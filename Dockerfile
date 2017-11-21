FROM bestwise/openface
COPY . /root/openface/websocket
WORKDIR /root/openface/websocket
RUN pip install --no-cache-dir --disable-pip-version-check -i https://mirrors.ustc.edu.cn/pypi/web/simple tornado

CMD ["python", "websocket.py"]