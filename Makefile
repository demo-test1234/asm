

build:
	docker build --platform linux/amd64 \
	  --build-arg HTTP_PROXY="http://192.168.111.111:1087" \
	  --build-arg HTTPS_PROXY="http://192.168.111.111:1087" \
	  --progress=plain \
	  -t aigcpanel-server-musetalk:latest .