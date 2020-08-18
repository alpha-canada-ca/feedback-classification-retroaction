(La version française suit la version anglaise)

# Docker

There are three folders:
- **both:** contains the docker recipe for a container that has both the CRON job for training and web service API for providing suggested tags
- **suggestion** contains the docker recipe for the web service that suggests tags
- **train** contains the docker recipe for the CRON job that retrains the models

## How to build 
- Ensure that docker and docker-compose are installed
- From that base folder of the project run: docker-compose build

## How to run locally
- run: docker-compose up
- go to localhost:5000/suggestCategory?lang=en&section=Health&text=Should I wear a mask
- You should see PPE in the browser window if using the default models

## If a new container version is needed
- edit docker-compose.yml from the base directory
- on line 4 change: image: "ryanhyma/suggestion:1.0.0" to a new version or update the dockerhub repo by assigning a new account
- run docker-compose build from the base folder


