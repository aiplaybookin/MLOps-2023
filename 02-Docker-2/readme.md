1. Use a shared volume called mnist for data exchange between services.
```
volumes:
  mnist:
```
Then use this volume across all services

2. If you want to inspect the volume from outside the container, you can use the docker volume command. Run the following command to inspect the "mnist" volume:
```
docker inspect mnist
```