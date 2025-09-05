가상환경 env_rcnn 생성 후 //
pip install -r requirements.txt

자사 코드(c++)에 아래와 같이 코드 적용

	std::string command = "C:/Users/CAMMSYS/anaconda3/envs/env_rcnn/python.exe C:/Users/CAMMSYS/Desktop/object_detection/fasterrcnn/inference_hsm.py";
	// Python 스크립트 호출
	int result = system(command.c_str());  // Python 스크립트 실행
	if (result == 0) {
	AfxMessageBox(_T("Python script executed successfully."));
	}
	else {
	AfxMessageBox(_T("Error executing Python script."));
	}


fasterrcnn 명령어 //

python train.py --model fasterrcnn_resnet50_fpn --epochs 100 --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4
