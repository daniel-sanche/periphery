.-PHONY: cluster deploy check-env

ZONE=us-central1-c
CLUSTER=ml-project
IMAGE_REPO=gcr.io/${PROJECT_ID}/${CLUSTER}

cluster: check-env
	gcloud container clusters create ${CLUSTER} \
	  --num-nodes 1 \
	  --accelerator type=nvidia-tesla-p100,count=1 \
	  --zone ${ZONE}
	skaffold run -p gpu --default-repo=${IMAGE_REPO} -l skaffold.dev/run-id=${CLUSTER}-${PROJECT_ID}-${ZONE}

deploy: check-env
	echo ${CLUSTER}
	gcloud container clusters get-credentials --project ${PROJECT_ID} ${CLUSTER} --zone ${ZONE}
	skaffold run -p gpu --default-repo=${IMAGE_REPO} -l skaffold.dev/run-id=${CLUSTER}-${PROJECT_ID}-${ZONE}

check-env:
ifndef PROJECT_ID
	$(error PROJECT_ID is undefined)
endif
