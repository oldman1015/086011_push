# -*- coding: utf-8 -*-
from flask import Flask, request, abort
from linebot import ( LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage ,ImageMessage,)
from skimage import io
import cv2
import sys,os,dlib,glob,numpy
# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 比對人臉圖片資料夾名稱
faces_folder_path = "./rec"

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()

# 載入人臉特徵點檢測器
sp = dlib.shape_predictor(predictor_path)

# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
candidate = []

# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    base = os.path.basename(f)
    # 依序取得圖片檔案人名
    candidate.append(os.path.splitext(base)[0])
    img = io.imread(f)

    # 1.人臉偵測
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        # 2.特徵點偵測
        shape = sp(img, d)
 
        # 3.取得描述子，128維特徵向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        # 轉換numpy array格式
        v = numpy.array(face_descriptor)
        descriptors.append(v)
if(len(descriptors)!=len(candidate)):
    print("某張照片無法辨識!!!!")
    exit()
app = Flask(__name__)

line_bot_api= LineBotApi('Ju1xLn6WMJPvSg8Fgk5/wYdd/TT0xX8YI29g4Ri2ccp1GtizaCMaKnKtTS/EccjtHtcVV7r1somm6K4NGy3kAAs/kJbYCiTbl5sxvAFxlHK18Do/Au3u71L9uhpkIcOYsweasRXygMJA/khctxjdTwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('604f60743f3c7267cd0bd21ed6f9cdb9')
line_bot_api.push_message("Ueafd60ea7a1999808ed1c91babf30d98",TextSendMessage(text="幹"))
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

# handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=[TextMessage,ImageMessage]) #當有收到訊息事件且訊息是文字格式則執行handle_message(),MsgTpe參考https://ithelp.ithome.com.tw/articles/10217402
def handle_message(event):#function name可以自定
    msg=""
    #print(event.source.user_id)
    if(event.message.type=="text"):
        msg=event.message.text
    if(event.message.type=="image"):
        message_content = line_bot_api.get_message_content(event.message.id)
        #filename=datetime.now()+".png"
        filename=event.source.user_id[-5:]+"_"+str(event.timestamp)+".png"
        #print(filename)

        #Ref:https://github.com/line/line-bot-sdk-python
        with open("./imgs/"+filename, 'wb') as fd:#Line收到的影像都會被轉成PNG??
            for chunk in message_content.iter_content(): 
                fd.write(chunk)
        frame = io.imread("./imgs/"+filename)
        dets = detector(frame, 1)
        #print("dets:")
        #print(dets)
        dist = []
        for k, d in enumerate(dets):
            shape = sp(frame, d)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            d_test = numpy.array(face_descriptor) # last face

            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            # 以方框標示偵測的人臉
            #cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2.LINE_AA)

        # 計算歐式距離
        #print(len(descriptors))
        for i in descriptors:
           dist_ = numpy.linalg.norm(i -d_test)
           dist.append(dist_)
           #print("--dist--")
           #print(dist)
        if len(dets) ==0:
            pass
            #cv2.imshow('Face Recognition',frame)
        else:
            # 將比對人名和比對出來的歐式距離組成一個dict
            c_d = dict( zip(candidate,dist))
            print("c_d:",c_d)
            # 根據歐式距離由小到大排序
            cd_sorted = sorted(c_d.items(), key = lambda d:d[1])

            print("cd_sorted",cd_sorted)

            # 取得最短距離就為辨識出的人名
            if(len(cd_sorted)):
                rec_name = cd_sorted[0][0]
            # 將辨識出的人名印到圖片上面
        msg="你的影像已收到..."+cd_sorted[0][0]
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=msg))

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=1031)
