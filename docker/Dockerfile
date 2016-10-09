FROM python:2.7

# Create a build directory
RUN mkdir -p /usr/src/myapp/build

# Copy the requirements and the bootsrap script
COPY bootstrap.sh /usr/src/myapp/build/bootstrap.sh
COPY requirements.txt /usr/src/myapp/build/requirements.txt
COPY extras.txt /usr/src/myapp/build/extras.txt

# Build and install all the requirements
RUN /usr/src/myapp/build/bootstrap.sh
