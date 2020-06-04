# Periphery

Computer vision containers for edge devices. Designed for hobby projects

[![Project Video](https://img.youtube.com/vi/ZaxkfWmv-fo/0.jpg)](https://www.youtube.com/watch?v=ZaxkfWmv-fo)


---

## Getting Started - Docker

1. Build and launch the frontend

```
  cd controllers/web
  docker build -t front ./
  docker run --rm -p 8080:8080 --network host -it front
```

2. Build and launch one or more models
```
  MODEL=yolo_v3
  cd models/$MODEL
  docker build -t $MODEL ./
  docker run --rm --network host -it $MODEL
```

3. Launch your web browser to `http://localhost:8080`

### Getting Started - Kubernetes

1. Set up a cluster

2. Deploy manifests using skaffold
```
  skaffold run
```

3. Port forward to the frontend deployment
```
  kubectl port-forward deployment/frontend 8080:8080
```

3. Launch your web browser to `http://localhost:8080`

---

## Controller Interface

The controller defines the application logic. The controller manages user input,
marshals data between containers, and displays the final output to the end user

### Web Socket Endpoints

- ***frame_request()***
  - called by models when they are ready for a new image to process
- ***frame_complete(annotations)***
  - called by models when processing is complete
- ***register(model_name)***
  - called by models when they join the network

## Model Interface

Models represent a single ML task. By default, models will continuously poll the
controller for new frames when idle, but can also be used on-demand if required

### Web Socket Endpoints
- ***process_frame(data_url)***
  - used to process an individual image

---

## YAML formats

### Model Output YAML

```
name: yolo
time: 0.5       # optional
inference_time: 0.3    # optional
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
  - kind: text
    data: "hello"
    confidence: 1.0
  - kind: mask
    points:
      - x: 10
        y: 10
    confidence: 1.0
    label: person
  - kind: lines
    points:
      - label: head
        x: 10
        y: 20
    links:
      - from: head
        to: neck
    confidence: 1.0
  - kind: points
    points:
      - name: head
        x: 10
        y: 20
        confidence: 0.7
```
