# simple-r


Build the docker image locally:
```
docker build -t changedetector . 
```

Upload the image locally:
```
tira-run \
	--image changedetector \
	--input-dataset multi-author-writing-style-analysis-2024/multi-author-spot-check-20240428-training \
	--tira-vm-id karami-sh \
	--push true
```
