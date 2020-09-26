# -*- coding: utf-8 -*-  
import cv2,sys,os,csv,datetime
from pathlib import Path,WindowsPath
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from time import sleep
import tkinter as tk
from tkinter import simpledialog,messagebox

class FaceCheck:
    def __init__(self):
        self.workPath=Path(sys.path[0])
        """工作路径"""
        self.dataPath=self.workPath/"data"
        self.classifierFile=self.workPath/"haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(str(self.classifierFile.resolve()))
        self.userFile=self.dataPath/"member.csv"
        self.names=[]
        self.cap=None
        self.loadUser()
    def loadUser(self):
        """加载识别人列表"""
        if self.userFile.exists():
            with open(self.userFile,"r",encoding='UTF-8') as csv_file:
                reader = csv.reader(csv_file)
                for item in reader:
                    self.names.append(item[0])
    def data_collection(self):
        """数据录入"""
        if(not self.dataPath.exists()):
            self.dataPath.mkdir()
        # 调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)#cv2.CAP_DSHOW是作为open调用的一部分传递标志，还有许多其它的参数，而这个CAP_DSHOW是微软特有的。
        face_name=tk.simpledialog.askstring('输入','请输入需要识别者的名字:')
        if not face_name:
            print("输入名字位空")
            return
        with open(self.userFile,'a',newline="",encoding='UTF-8')as f:
                writer = csv.writer(f)
                writer.writerow([face_name])
                self.names.append(face_name)
        print('数据初始化中，请直视摄像机录入数据....')
        count = 0
        _output = sys.stdout
        while True:
            # 从摄像头读取图片
            sucess, img = cap.read()
            # 转为灰度图片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)#1.image表示的是要检测的输入图像# 2.objects表示检测到的人脸目标序列# 3.scaleFactor表示每次图像尺寸减小的比例
            face_id=len(self.names)
            datapath=str(self.dataPath.resolve())
            for (x, y, w, h) in faces:
                #画矩形
                cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 0))
                count += 1
                # 保存图像
                cv2.imwrite(datapath+"/Member." + str(face_id) + '.' + str(count) + '.jpg', gray[y: y + h, x: x + w])
                cv2.imshow('data collection', img)
            _output.write(f'\r获取数据:{count:.0f}')
            # 将标准输出一次性刷新
            _output.flush()
            # 保持画面的持续。
            k = cv2.waitKey(1)
            if k == 27:  # 通过esc键退出
                break
            elif count >= 200:  # 得到n个样本后退出摄像
                break
        cap.release()
        cv2.destroyAllWindows()
        tk.messagebox.showinfo("提示","收集完毕")
    def face_training(self):
        """数据训练"""
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        #LBP是一种特征提取方式，能提取出图像的局部的纹理特征
        def get_images_and_labels(path):
            if not path.exists():
                path.mkdir()
                return
            faceSamples = []
            ids = []
            # 遍历图片路径，导入图片和id，添加到list
            for imagePath in path.glob("*.jpg"):
                PIL_img = Image.open(imagePath).convert('L') #通过图片路径并将其转换为灰度图片。
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = self.face_detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x: x + w])
                    ids.append(id)
            return faceSamples, ids

        faces, ids = get_images_and_labels(self.dataPath)
        if len(ids)==0:
            tk.messagebox.showinfo("提示","先录入脸部数据")
            return
        print('数据训练中')
        recognizer.train(faces, np.array(ids))
        recognizer.write(str(self.dataPath/'trainer.yml'))
        tk.messagebox.showinfo("提示","训练完成")
    def face_check(self):
        cap = cv2.VideoCapture(0)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        recognizer.read(str(self.dataPath/'trainer.yml'))
        faceCascade = cv2.CascadeClassifier(str(self.classifierFile))
        idnum = 0
        cam = cv2.VideoCapture(0)
        #设置大小
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            ret, img = cam.read()
            #图像灰度处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,#表示在前后两次相继的扫描中，搜索窗口的比例系数
                minNeighbors=5,#表示构成检测目标的相邻矩形的最小个数(默认为3个)
                minSize=(int(minW), int(minH))#minSize和maxSize用来限制得到的目标区域的范围
            )
        
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 返回侦测到的人脸的id和近似度conf（数字越大和训练数据越不像）
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if confidence<80:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("校验成功")
                    return "success"
                k = cv2.waitKey(1)
                if k == 27:  # 通过esc键退出
                    print("校验失败，手动退出")
                    break
    def face_ientification(self):
        """脸部识别"""
        if len(self.names)==0:
            tk.messagebox.showinfo("提示","先录入脸部数据")
            return
        cap = cv2.VideoCapture(0)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        recognizer.read(str(self.dataPath/'trainer.yml'))
        faceCascade = cv2.CascadeClassifier(str(self.classifierFile))
        font = ImageFont.truetype("simsun.ttc",30)

        idnum = 0
        cam = cv2.VideoCapture(0)
        #设置大小
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)
        def zh_ch(string):
            return string.encode('GBK').decode(errors='ignore')

        while True:
            ret, img = cam.read()
            #图像灰度处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,#表示在前后两次相继的扫描中，搜索窗口的比例系数
                minNeighbors=5,#表示构成检测目标的相邻矩形的最小个数(默认为3个)
                minSize=(int(minW), int(minH))#minSize和maxSize用来限制得到的目标区域的范围
            )
            namess=""
            confidence=None
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 返回侦测到的人脸的id和近似度conf（数字越大和训练数据越不像）
                idnum, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                idnum-=1
                if confidence < 100:
                    namess = self.names[idnum]
                    confidence = "{0}%".format(round(100 - confidence))
                else:
                    namess = "unknown"
                    confidence = "{0}%".format(round(100 - confidence))
                if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img)
                draw.text((x + 5, y - 6), str(namess), font=font, fill=(0, 0, 255))
                img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)#输出置信度

            cv2.imshow(zh_ch('image'), img)
            k = cv2.waitKey(5)
            if k == 13:#回车
                theTime = datetime.datetime.now()
                strings = [str(self.names[idnum]),str(confidence),str(theTime)]
                print(strings)
                with open(self.dataPath/"log.csv", "a",newline="",encoding='UTF-8') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([str(self.names[idnum]),str(confidence),str(theTime)])
            elif k==27:#ESC退出
                cap.release()
                cv2.destroyAllWindows()
                break
    def face_clean(self):
        """清除数据"""
        if(self.dataPath.exists()):
            if not hasattr(WindowsPath,'clean'):
                def clean(self):
                    """递归删除目录"""
                    for item in self.iterdir():
                        if item.is_file():
                            item.unlink()
                            continue
                        if item.iterdir():
                            abs_path = self.joinpath(item.name)
                            abs_path.clean()
                            item.rmdir()
                WindowsPath.clean=clean
            self.dataPath.clean()
        self.__init__()
        return
    def listUser(self):
        tk.messagebox.showinfo("提示","".join([str(i+1)+".  "+x for i,x in enumerate(self.names)]))
    def panel(self):
        """界面选择"""
        self.root = tk.Tk()
        self.root.wm_attributes('-topmost',1)
        self.root.title('脸部识别')
        list = tk.Button(self.root, text ="查看录入列表", command = self.listUser)
        collection = tk.Button(self.root, text ="录入人脸", command = self.data_collection)
        training = tk.Button(self.root, text ="训练模型", command = self.face_training)
        ientification = tk.Button(self.root, text ="脸部识别(ESC退出 Enter打印日志)", command = self.face_ientification)
        check = tk.Button(self.root, text ="脸部校验(ESC退出)", command = self.face_check)
        clean = tk.Button(self.root, text ="清除数据", command = self.face_clean)
        
        list.pack()
        collection.pack()
        training.pack()
        ientification.pack()
        check.pack()
        clean.pack()
        # self.root.protocol('WM_DELETE_WINDOW', on_closing)
        self.root.mainloop()
def open_check():
    return FaceCheck().face_check()
if __name__ == '__main__':
    FaceCheck().panel()