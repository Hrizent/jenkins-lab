
# download
python3 -m venv ./my_env
. ./my_env/bin/activate
ls -la # debug
python3 -m pip install --upgrade pip
python3 -m pip install mlflow pandas numpy scikit-learn
python3 download.py

# train
echo "Start train model"
cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate
python3 train_model.py

# deploy
cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate   #активировать виртуальное окружение

export BUILD_ID=dontKillMe            #параметры для jenkins чтобы не убивать фоновый процесс для mlflow сервиса
export JENKINS_NODE_COOKIE=dontKillMe #параметры для jenkins чтобы не убивать фоновый процесс для mlflow сервиса
path_model=$(cat best_model.txt) #прочитать путь из файла в bash переменную 
mlflow models serve -m $path_model -p 5003 --no-conda & #запуск mlflow сервиса на порту 5003 в фоновом режиме

# healthy
curl http://127.0.0.1:5003/invocations -H"Content-Type:application/json"  --data '{"inputs": [[ -1.75938045, -1.2340347 , -1.41327673,  0.76150439,  2.20097247, -0.10937195,  0.58931542,  0.1135538]]}'


# pipeline
pipeline {
    agent any

    stages {
        stage('Start Download') {
            steps {
                
                build job: "download"
                
            }
        }
        
        stage ('Train') {
            
            steps {
                
                script {
                    dir('/var/lib/jenkins/workspace/download') {
                        build job: "train"
                    }
                }
            
            }
        }
        
        stage ('Deploy') {
            steps {
                build job: 'deploy'
            }
        }
        
        stage ('Status') {
            steps {
                build job: 'healthy'
            }
        }
    }
}
