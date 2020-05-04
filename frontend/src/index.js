const app = require('express')();
const http = require('http').createServer(app);
const io = require('socket.io')(http);

// returns a simple response 
app.get('/', (req, res) => {
  console.log(`received request: ${req.method} ${req.url}`)
  res.sendFile(__dirname + '/index.html');
});

// starts an http server on the $PORT environment variable
const PORT = process.env.PORT || 8080;
http.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
  console.log('Press Ctrl+C to quit.');
});

// quit on Ctrl+C
process.on('SIGINT', function() {
    process.exit();
});

// web socket connection
io.on('connection', (socket) => {
  console.log('a user connected');


  socket.on('disconnect', () => {
    console.log('user disconnected');
  });

  socket.on('chat message', (msg) => {
    console.log('message: ' + msg);
    io.emit('chat message', msg);
  });
});

module.exports = app
