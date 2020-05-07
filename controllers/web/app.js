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

  socket.on('frame_complete',function(yolo_dict){
    socket.broadcast.emit('render_update', yolo_dict);
  });

  socket.on('frame_request',function(){
    socket.broadcast.emit('webcam_request', socket.id);
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
