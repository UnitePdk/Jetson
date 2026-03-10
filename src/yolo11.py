from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    # 새로운 모델 학습시키기
    model = YOLO("yolo26n.pt") #yolo v11을 사용해서 데이터 학습
    results = model.train(data="datasets/data.yaml", epochs=30, imgsz=640, batch=8) # 데이터셋 정보가 담겨있는 data.yaml 위치 지정
    metrics = model.val()
    print("학습 수준:")
    print(metrics.box.map50)  # 0~1까지 학습 완성도를 표시


    # 이미 학습시킨 모델 불러와서 사용하기
    # model = YOLO("runs/detect/train/weights/best.pt") # 학습된 모델의 경로를 잘 입력

    # 이미지 추론
    result = model("drone1.png")

    # 결과 이미지 그리기
    img_with_boxes = result[0].plot()  # plot()은 NumPy 배열 반환

    # 이미지 크기 조절 (예: 너비 960px 기준 비율 축소)
    scale_width = 960
    scale_ratio = scale_width / img_with_boxes.shape[1]
    resized_img = cv2.resize(img_with_boxes, (0, 0), fx=scale_ratio, fy=scale_ratio)

    # 이미지 출력
    cv2.imshow("YOLO Detection (Resized)", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()