FROM python:2.7

# Create a build directory
RUN mkdir -p /usr/src/myapp/build

# Copy the requirements and the bootsrap script
COPY bootstrap.sh /usr/src/myapp/build/bootstrap.sh
COPY requirements.txt /usr/src/myapp/build/requirements.txt

# Build and install all the requirements
RUN /usr/src/myapp/build/bootstrap.sh

# Set the right environment variables
ENV PYTHONPATH /usr/share/pyshared/:/usr/lib/pyshared/python2.7/vtk/
ENV PETSC_DIR /usr/lib/petscdir/3.4.2/
ENV PETSC_ARCH linux-gnu-c-opt

# Apparently TINI will prevent kernel crahses
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

WORKDIR /usr/src/myapp

EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
