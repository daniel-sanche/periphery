export const name = 'main'

import Vue from 'vue';
import socketio from 'socket.io';
import VueSocketIO from 'vue-socket.io';

export const SocketInstance = socketio('http://localhost:4113');

Vue.use(VueSocketIO, SocketInstance)

