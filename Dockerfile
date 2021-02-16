FROM fedora:32
LABEL maintainer="steve.kuznetsov@gmail.com"

RUN dnf install -y python3 python3-devel python3-pip gcc-c++ gdal gdal-devel CGAL-devel coin-or-Cbc
RUN pip install pipenv
ADD Pipfile Pipfile.lock /terrain/
WORKDIR /terrain/
RUN pipenv install