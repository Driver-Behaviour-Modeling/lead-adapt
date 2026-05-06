mkdir -p 3rd_party/CARLA_0915

cd 3rd_party/CARLA_0915

wget -O CARLA_0915.tar.gz https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvzf CARLA_0915.tar.gz
cd Import
wget -O AdditionalMaps_0.9.15.tar.gz https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
tar -xvzf AdditionalMaps_0.9.15.tar.gz
cd ..
bash ImportAssets.sh
