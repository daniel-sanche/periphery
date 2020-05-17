.-PHONY: cluster deploy check-env port-forward

ZONE=us-central1-c
CLUSTER=ml-project
IMAGE_REPO=gcr.io/${PROJECT_ID}/${CLUSTER}

cluster: check-env
	gcloud container clusters create ${CLUSTER} --zone ${ZONE}
	gcloud container node-pools create gpu-pool \
	  --cluster ${CLUSTER} \
	  --zone ${ZONE} \
	  --num-nodes 1 \
	  --accelerator type=nvidia-tesla-p100,count=1
	skaffold run -p gpu --default-repo=${IMAGE_REPO} -l skaffold.dev/run-id=${CLUSTER}-${PROJECT_ID}-${ZONE}

deploy: check-env
	echo ${CLUSTER}
	gcloud container clusters get-credentials --project ${PROJECT_ID} ${CLUSTER} --zone ${ZONE}
	skaffold run -p gpu --default-repo=${IMAGE_REPO} -l skaffold.dev/run-id=${CLUSTER}-${PROJECT_ID}-${ZONE}

port-forward:
	kubectl port-forward deployment/frontend 8081:8080

check-env:
ifndef PROJECT_ID
	$(error PROJECT_ID is undefined)
endif
