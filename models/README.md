# Model Contract
- designed to be composable and scaleable
- typically stateless
- environment variables are set at deploy time, but configurable at runtime
- can process data in two ways:
  - push: controller sends image to model through process_frame()
  - pull: AUTO_RUN is enabled and model will continuously find new frames

## Web Sockets
- process_frame()
- update_variable(var_name, new_value)
  - change settings dynamically

## Environment Variables
- CONTROLLER_URI
  - the location of the controller to communicate with
- AUTO_RUN
  - when enabled, will continuously query CONTROLLER_URI/frame_request when idle
- MAX_FPS
  - introduce delays between frames if there's extra capacity
