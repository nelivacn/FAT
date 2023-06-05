#!/bin/sh

SHELL_FOLDER=$(dirname $(readlink -f $0))
cd $SHELL_FOLDER
APP_NAME='docker-ceping-web'
INSTANCE_NAME="${APP_NAME}-1"
GREP_KEY="Dmyname=$INSTANCE_NAME"
APP_FOLDER="${SHELL_FOLDER}"
OUT_FILE="${APP_FOLDER}/OUT"

JAR_FILE="$APP_FOLDER/$APP_NAME-23.5.6-ALPHA.jar"

JAVA_HOME="${APP_FOLDER}/jre"


SPRING_PROFILE='prod,company'

REMOTE_DEBUG=''

run_java(){
  nohup $JAVA_HOME/bin/java \
  -Xms200m -Xmx500m -XX:MetaspaceSize=150m \
  -$GREP_KEY \
  -Duser.timezone=Asia/Shanghai \
  $REMOTE_DEBUG \
  -jar $JAR_FILE \
  --spring.profiles.active=$SPRING_PROFILE \
  1>$OUT_FILE 2>&1 &
}

start0(){
  is_exist
  if [ $? -eq 0 ];then
    echo "$INSTANCE_NAME is already running. pid=$PID."
    return 0
  else
    run_java
    sleep 3
    is_exist
    if [ $? -eq 0 ];then
      echo "$INSTANCE_NAME is running now. pid=$PID."
    else
      echo "$INSTANCE_NAME run failed."
    fi
  fi
}

is_exist(){
  PID=`ps -ef|grep -v "grep" |grep -w $GREP_KEY|awk '{print $2}'`
  if [ -z "$PID" ];then
    return 1
  else
    return 0
  fi
}

status(){
  is_exist
  if [ $? -eq 0 ];then
    echo "$INSTANCE_NAME is running. pid=$PID."
  else
    echo "$INSTANCE_NAME is not running."
  fi
}

stop1(){
  is_exist
  if [ $? -eq 0 ];then
    kill -15 ${PID}
    if [ $? -ne 0 ];then
      echo "stop $INSTANCE_NAME failed!"
      return 1
    else
      is_exist
      while [ $? -eq 0 ]
      do
        sleep 1
        echo '.'
        is_exist
      done	
      echo "$INSTANCE_NAME stopped."
      return 0
    fi
  else
    echo "$INSTANCE_NAME is not running!"
    return 1
  fi
}

restart(){
  stop1
  sleep 3
  start0
}

usage(){
  echo "usage: $0 {start|restart|stop|status}"
}

case "$1" in
  'start')
    start0
    ;;
  'stop')
    stop1
    ;;
  'restart')
    restart
    ;;
  'status')
    status
    ;;
  *)
    usage
    ;;
esac
