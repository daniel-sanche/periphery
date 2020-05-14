var express = require("express");
var app = new express ();
var http = require("http").Server(app);
var io = require("socket.io")(http);

var connected_clients = {};

app.use(express.static(__dirname + "/public"));

app.get('/',function(req,res){
  res.redirect('index.html');
});

io.on('connection',function(socket){
  console.log('a user connected');

  socket.on('disconnect', () => {
    kind = connected_clients[socket.id];
    console.log(kind + ' disconnected');
    socket.to('client').emit('model_disconnect', kind);
    delete connected_clients[socket.id];
  });

  socket.on('register', (kind) => {
    console.log(kind + ' registered');
    socket.join(kind);
    connected_clients[socket.id] = kind;
  });

  socket.on('frame_complete',function(yolo_dict){
    socket.to('client').emit('render_update', yolo_dict);
  });

  socket.on('frame_request',function(){
    model_name = connected_clients[socket.id];
    socket.to('client').emit('webcam_request', socket.id, model_name);
  });

  socket.on('webcam_response',function(requester_id, image_data){
    socket.to(requester_id).emit('process_frame', image_data);
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
