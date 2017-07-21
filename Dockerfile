FROM altermarkive/lab-environment

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
RUN pip3 install runipy==0.1.3 && sed -i "s/'svg'\,/'svg',\ 'text\/vnd\.plotly\.v1\+html'\:\ 'html'\,/g" /usr/local/lib/python3.5/dist-packages/runipy/notebook_runner.py
RUN pip install plotly==2.0.12
RUN git clone https://github.com/compmonks/SOMPY.git && cd /SOMPY && git reset --hard 5e94ba6d2d5457b3ae231eabed9b63436639d13d
RUN cd /SOMPY && sed -i 's/,ipdb//g' sompy/sompy.py && sed -i 's/import\ ipdb//g' sompy/visualization/mapview.py && python3 setup.py install

ADD *.py /code/
ADD models /code/models
ADD notebooks /code/notebooks
ADD bh_tsne /code/bh_tsne

RUN cd /code/bh_tsne && g++ sptree.cpp tsne.cpp -o bh_tsne -O2 2> /dev/null

CMD ["/usr/bin/python3", "/code/run.py"]
