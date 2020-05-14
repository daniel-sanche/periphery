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
      console.log(result_dict)
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
    color_for_model(model_name){
      // blue red orange green
      const colors = ['#0328fc', '#fc0317',  '#fca903', '#5afc03'];
      var idx = this.connected_models.indexOf(model_name);
      return colors[idx % colors.length];
    },

    draw_border(ctx, color){
        ctx.beginPath();
        ctx.lineWidth = "5";
        ctx.strokeStyle = color;
        ctx.rect(0, 0, this.width, this.height);
        ctx.stroke();
    },

    color_shift(p,c0,c1,l) {
        let r,g,b,P,f,t,h,i=parseInt,m=Math.round,a=typeof(c1)=="string";
        if(typeof(p)!="number"||p<-1||p>1||typeof(c0)!="string"||(c0[0]!='r'&&c0[0]!='#')||(c1&&!a))return null;
        if(!this.pSBCr)this.pSBCr=(d)=>{
            let n=d.length,x={};
            if(n>9){
                [r,g,b,a]=d=d.split(","),n=d.length;
                if(n<3||n>4)return null;
                x.r=i(r[3]=="a"?r.slice(5):r.slice(4)),x.g=i(g),x.b=i(b),x.a=a?parseFloat(a):-1
            }else{
                if(n==8||n==6||n<4)return null;
                if(n<6)d="#"+d[1]+d[1]+d[2]+d[2]+d[3]+d[3]+(n>4?d[4]+d[4]:"");
                d=i(d.slice(1),16);
                if(n==9||n==5)x.r=d>>24&255,x.g=d>>16&255,x.b=d>>8&255,x.a=m((d&255)/0.255)/1000;
                else x.r=d>>16,x.g=d>>8&255,x.b=d&255,x.a=-1
            }return x};
        h=c0.length>9,h=a?c1.length>9?true:c1=="c"?!h:false:h,f=this.pSBCr(c0),P=p<0,t=c1&&c1!="c"?this.pSBCr(c1):P?{r:0,g:0,b:0,a:-1}:{r:255,g:255,b:255,a:-1},p=P?p*-1:p,P=1-p;
        if(!f||!t)return null;
        if(l)r=m(P*f.r+p*t.r),g=m(P*f.g+p*t.g),b=m(P*f.b+p*t.b);
        else r=m((P*f.r**2+p*t.r**2)**0.5),g=m((P*f.g**2+p*t.g**2)**0.5),b=m((P*f.b**2+p*t.b**2)**0.5);
        a=f.a,t=t.a,f=a>=0||t>=0,a=f?a<0?t:t<0?a:a*P+t*p:0;
        if(h)return"rgb"+(f?"a(":"(")+r+","+g+","+b+(f?","+m(a*1000)/1000:"")+")";
        else return"#"+(4294967296+r*16777216+g*65536+b*256+(f?m(a*255):0)).toString(16).slice(1,f?undefined:-2)
    },

    render_annotations(annotations_data,  model_name){
      const vm = this;
      const canvas = document.getElementById(model_name);
      const color = this.color_for_model(model_name);
      if (canvas) {
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (vm.enabled_dict[model_name]) {
          for (var obj_idx=0; obj_idx<annotations_data.length; obj_idx++) {
            const obj = annotations_data[obj_idx];
            if (obj.kind == 'box'){
              ctx.beginPath();
              ctx.lineWidth = "2";
              ctx.strokeStyle = color;
              ctx.rect(obj.x, obj.y, obj.width, obj.height);
              ctx.stroke();
              ctx.font = "20px Arial";
              labelText = obj.label + " - " + obj.confidence.toFixed(2);
              var labelX = Math.min(Math.max(obj.x, 0), canvas.width-ctx.measureText(labelText).width);
              var labelY = Math.max(obj.y, 20);
              ctx.strokeText(labelText, labelX, labelY);
            } else if (obj.kind == 'image'){
              console.log(obj.data);
              ctx.drawImage(this.img, 0, 0);
              vm.img.src = obj.data;
              vm.img.onload = function() {
                ctx.drawImage(vm.img, 0, 0);
                vm.draw_border(ctx, color);
              }
              vm.draw_border(ctx, color);
            } else if (obj.kind == 'mask'){
              coefficient = 1;
              if (obj_idx % 2 == 0){
                coefficient = -1;
              }
              // alpha = ~60%
              alpha = "90"
              mask_color = vm.color_shift(coefficient * 0.25 * obj_idx, color + alpha);
              ctx.beginPath();
              ctx.fillStyle = mask_color;
              for (var i=0; i<obj.points.length; i++){
                if (i == 0){
                  ctx.moveTo(obj.points[i].x, obj.points[i].y);
                } else {
                  ctx.lineTo(obj.points[i].x, obj.points[i].y);
                }
              }
              ctx.closePath();
              ctx.fill();
              ctx.strokeStyle = color;
              ctx.fillStyle = color;
              ctx.font = "20px Arial";
              labelText = obj.label + " - " + obj.confidence.toFixed(2);
              var labelX = Math.min(Math.max(obj.points[0].x, 0), canvas.width-ctx.measureText(labelText).width);
              var labelY = Math.max(obj.points[0].y, 20);
              ctx.strokeText(labelText, labelX, labelY);
            }
          }
        }
      }
    }
  }
});
