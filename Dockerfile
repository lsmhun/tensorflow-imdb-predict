FROM python:3.6.7

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . /app

# Install production dependencies.
RUN pip install -e . && pip install --trusted-host pypi.python.org -Ur requirements.txt
#EXPOSE $PORT
EXPOSE 5000

CMD ["python", "lsmhun/flask_app.py"]

	
	
	

