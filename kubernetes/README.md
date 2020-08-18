(La version française suit la version anglaise)

# Kubernetes
The kubernetes scripts are setup for a small cluster

There are three files:
- **suggestion-deployment:** creates a deployment resource in kubernetes and run the suggestion container
- **suggestion-service** exposes the deployment to a service
- **suggetsion-ingress** maps the service to a URL, and also contains logic for cert-manager and automatic cert generation

## How to deploy initially and manually
- Install kubectl and setup with your cluster
- Create a namespace for the service
- edit suggestion-deployment and update the image section with the tag from docker-compose
- kubectl apply -f suggetion-deployment.yaml
- kubectl apply -f suggestion-service.yaml
- kubectl apply -f suggestion-ingress.yaml


## How to run locally
- run: docker-compose up
- go to localhost:5000/suggestCategory?lang=en&section=Health&text=Should I wear a mask
- You should see PPE in the browser window if using the default models

## If a new container version is needed
- edit docker-compose.yml from the base directory
- on line 4 change: image: "ryanhyma/suggestion:1.0.0" to a new version or update the dockerhub repo by assigning a new account
- run docker-compose build from the base folder


