var express = require("express");
var app = new express ();
var http = require("http").Server(app);
var io = require("socket.io")(http);

app.use(express.static(__dirname + "/public"));

app.get('/',function(req,res){
  res.redirect('index.html'); //para archivos estaticos
});

io.on('connection',function(socket){
  console.log('a user connected');

  socket.on('disconnect', () => {
    console.log('user disconnected');
  });

  socket.on('camera',function(image){
    console.log(`new image`);
    socket.broadcast.emit('stream',image);
  });

  socket.on('yolo',function(yolo_dict){
    console.log(`new yolo`);
  });

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

// https://github.com/Jeixonx/webcam-stream-nodejs-socketio
