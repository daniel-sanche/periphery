var socket = io();

socket.on('connect', () => {
  console.log('connected');
  socket.emit('register', 'client');
});

var app = new Vue({
  el: '#vue',
  data: {
    height: 480,
    width: 640,
    connected_models: [],
    latest_states: {},
    img: new Image,
    enabled_dict: {}
  },

  created: function() {
    var vm = this;
    var render_canvas = document.getElementById("renderer");
    var render_context = render_canvas.getContext("2d");
    render_canvas.width = this.width;
    render_canvas.height = this.height;
    render_context.width = this.width;
    render_context.height = this.height;

    socket.on('webcam_request', function(requester_id){
      if (video.srcObject){
        render_context.drawImage(video, 0,0,render_context.width, render_context.height);
        socket.emit('webcam_response', requester_id, render_canvas.toDataURL('image/webp'));
      } else {
        console.log('camera not authorized');
      }
    });

    socket.on('render_update', function(result_dict){
      model_name = result_dict.name;
      if (!(vm.connected_models.includes(model_name))){
        vm.connected_models.push(model_name);
        vm.enabled_dict[model_name] = true;
      }
      vm.latest_states[model_name] = result_dict;
      vm.render_annotations(result_dict.annotations, model_name);
    });

    socket.on('model_disconnect', (model_name) => {
      const idx = vm.connected_models.indexOf(model_name);
      if (idx > -1) {
        vm.connected_models.splice(idx, 1);
      }
    });

    socket.on('disconnect', () => {
      console.log('server down');
      vm.connected_models = [];
    });
  },

  mounted() {
    var video = document.getElementById("video");
    const constraints = { video: true };
    navigator.getUserMedia= (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msgGetUserMedia);
    if(navigator.getUserMedia) {
      navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
      });
    }
  },

  methods: {
    render_annotations(annotations_data,  model_name){
      var vm = this;
      var canvas = document.getElementById(model_name);
      if (canvas) {
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (vm.enabled_dict[model_name]) {
          for (let obj of annotations_data) {
            if (obj.kind == 'box'){
              ctx.beginPath();
              ctx.lineWidth = "2";
              ctx.strokeStyle = "red";
              ctx.rect(obj.x, obj.y, obj.width, obj.height);
              ctx.stroke();
              ctx.font = "20px Arial";
              labelText = obj.label + " - " + obj.confidence.toFixed(2);
              var labelX = Math.min(Math.max(obj.x, 0), canvas.width-ctx.measureText(labelText).width);
              var labelY = Math.max(obj.y, 20);
              ctx.strokeText(labelText, labelX, labelY);
            } else if (obj.kind == 'image'){
              ctx.drawImage(this.img, 0, 0);
              vm.img.src = obj.data;
              vm.img.onload = function() {
                ctx.drawImage(vm.img, 0, 0);
              }
              canvas.style.zIndex = 1;
            }
          }
        }
      }
    }
  }
});
