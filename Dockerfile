FROM altermarkive/lab-environment

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

ADD *.py /code/
ADD models /code/models
ADD notebooks /code/notebooks
ADD bh_tsne /code/bh_tsne

RUN cd /code/bh_tsne && g++ sptree.cpp tsne.cpp -o bh_tsne -O2 2> /dev/null

CMD ["/usr/bin/python3", "/code/run.py"]
