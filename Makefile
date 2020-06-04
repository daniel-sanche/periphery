.-PHONY: cluster deploy check-env port-forward

ZONE=us-west1-b
CLUSTER=periphery
IMAGE_REPO=gcr.io/${PROJECT_ID}/${CLUSTER}

cluster: check-env
	gcloud container clusters create ${CLUSTER} --zone ${ZONE} --num-nodes 1
	gcloud container node-pools create gpu-pool \
	  --cluster ${CLUSTER} \
	  --zone ${ZONE} \
	  --num-nodes 4 \
	  --accelerator type=nvidia-tesla-k80,count=1
	skaffold run -p gpu --default-repo=${IMAGE_REPO} -l skaffold.dev/run-id=${CLUSTER}-${PROJECT_ID}-${ZONE}

deploy-gke: check-env
	echo ${CLUSTER}
	gcloud container clusters get-credentials --project ${PROJECT_ID} ${CLUSTER} --zone ${ZONE}
	kubectl delete deployments --all
	skaffold run -p gpu --default-repo=${IMAGE_REPO} -l skaffold.dev/run-id=${CLUSTER}-${PROJECT_ID}-${ZONE}

deploy:
	kubectl delete deployments --all
	skaffold run -l skaffold.dev/run-id=${CLUSTER}-${PROJECT_ID}-${ZONE}

port-forward:
	kubectl port-forward deployment/frontend 8080:8080

check-env:
ifndef PROJECT_ID
	$(error PROJECT_ID is undefined)
endif
