# Set the retention time to 10 hours
sudo docker exec -it broker kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic queries --config retention.ms=36000000
