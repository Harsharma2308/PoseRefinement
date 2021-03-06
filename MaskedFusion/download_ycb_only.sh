if [ ! -d datasets/ycb/YCB_Video_Dataset ];then
echo 'Downloading the YCB-Video Dataset'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi" -O YCB_Video_Dataset.zip && rm -rf /tmp/cookies.txt \
&& unzip YCB_Video_Dataset.zip \
&& mv YCB_Video_Dataset/ datasets/ycb/ \
&& rm YCB_Video_Dataset.zip
fi
