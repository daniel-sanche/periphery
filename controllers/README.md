# Controller Contract
- define the "application" made up of multiple jobs
- manages user input, marshals data between jobs, displays end result


## Web Sockets
- frame_request()
  - when called, will return a new frame to the caller
- frame_complete (result_yaml)
  - called when a job is finished processing a frame
- register(intro_yaml)
  - job introduces itself to the controller
  - gives a list of controlable parameters


### ResultYAML
```
name: yolo
annotations:
  - kind: box
    x: 0
    y: 0
    height: 100
    width: 100
    label: person
    confidence: 1.0
  - kind: image
    data: <b64 encoded image>
  - kind: string
    data: "hello"
process_time: 10
```

### ModelYAML
```
model_name: yolo
env:
  - name: MAX_FPS
    value: 60
```
