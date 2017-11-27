# coding:utf-8
import os
import io
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(fileDir, "..", ".."))

import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.options, tornado.httpclient
import tornado.gen

import tornado.websocket
import argparse
import StringIO
import cv2
import json
from PIL import Image
import numpy as np
import os
import base64
import urllib
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import openface
# 代配置
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')


args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            ('/soc', SocketHandler),
            ('/', IndexHandler)
        ]
        settings = dict(
            # login_url = "/auth/login",
            debug=True,
        )
        super(Application, self).__init__(handlers, **settings)


# websocket
class SocketHandler(tornado.websocket.WebSocketHandler):
    clients = set()
    # 默认为没有在训练模式
    # 跨域

    def check_origin(self, origin):
        #parsed_origin = urllib.parse.urlparse(origin)
        #return parsed_origin.netloc.endswith(".mydomain.com")
        return True

    def open(self):
        self.people = []
        self.allimages = []
        self.svm = None
        print("WebSocket connection open.")

    def on_message(self, message):
        msg = json.loads(message)
        # 添加人名
        if msg['type'] == "PREVIEW":
            self.display(msg['image'])
        # 训练的时候处理头像和人名
        elif msg['type'] == "TRAINING":
            self.collectfeature(msg['image'], msg['data'])
        # 增加人
        elif msg['type'] == "ADD_PERSON":
            if msg['data'].encode('ascii', 'ignore') in self.people:
                pass
            else:
                self.people.append(msg['data'].encode('ascii', 'ignore'))
            msg = {
                "type": "ADD_PERSON",
                "data": msg['data'].encode('ascii', 'ignore')
            }
            self.write_message(json.dumps(msg))
        elif msg['type'] == 'COLLECTED':
            if len(msg['data']) <= 1:
                self.svm = None
            else:
                self.trainSVM()
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def on_close(self):
        print("WebSocket connection closed.")

    def getData(self):
        X = []
        y = []
        # 遍历存储的人
        for img in self.allimages:
            X.append(img['rep'])
            y.append(img['name'])

        numIdentities = len(set(y + [-1])) - 1
        # 表示没有人
        if numIdentities == 0:
            return None

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def trainSVM(self):
        d = self.getData()
        if d is None:
            self.svm = None
            return
        else:
            (X, y) = d
            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            self.svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)

    def collectfeature(self, dataURL, identity):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        arryimage = Image.open(io.BytesIO(imgdata))
        # 镜像，应该还不需要镜像
        # rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        buf = np.asarray(arryimage)
        plt.figure()
        plt.imshow(buf)
        plt.xticks([])
        plt.yticks([])

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        plt.close()
        rgbFrame = cv2.cvtColor(np.array(arryimage), cv2.COLOR_BGR2RGB)
        # 获取屏幕中所有的脸
        # bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        # 训练时只取一张
        bb = align.getLargestFaceBoundingBox(rgbFrame)
        if bb is None:
            msg = {
                "type": "TRAINING",
                "data": {
                    "name": identity,
                    "image": content,
                    "success": False
                }
            }
            self.write_message(json.dumps(msg))
            return
        else:
            bbs = [bb]
        for bb in bbs:
            # print(len(bbs))
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                msg = {
                    "type": "TRAINING",
                    "data": {
                        "name": identity,
                        "image": content,
                        "success": False
                    }
                }
                self.write_message(json.dumps(msg))
                continue
            rep = net.forward(alignedFace)
            # 存储本次senssion的人名和特征
            self.allimages.append({"name": identity, "rep": rep})
            msg = {
                "type": "TRAINING",
                "data": {
                    "name": identity,
                    "image": content,
                    "success": True
                }
            }
            self.write_message(json.dumps(msg))

    def display(self, dataURL):
        head = "data:image/jpeg;base64,"
        assert (dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        # arryimage = Image.open(io.BytesIO(img))
        # 镜像，应该还不需要镜像,显示需要镜像
        # buf = np.fliplr(np.asarray(arryimage))
        buf = np.asarray(img)
        annotatedFrame = np.copy(buf)
        # rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame = cv2.cvtColor(np.array(buf), cv2.COLOR_BGR2RGB)
        # 获取屏幕中所有的脸
        bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        # 训练时只取一张
        # bb = align.getLargestFaceBoundingBox(rgbFrame)
        # bbs = [bb] if len(bb)bb is not None else []
        for bb in bbs:
            # print(len(bbs))
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue
            rep = net.forward(alignedFace)
            left = (bb.left(), bb.bottom())
            right = (bb.right(), bb.top())
            if len(self.people) == 0:
                name = "Unknown"
            elif len(self.people) == 1:
                name = self.people[0]
            elif self.svm:
                name = self.svm.predict(rep.reshape(1, -1))[0]
            else:
                print "hehehe"
                name = "Unknown"
            cv2.rectangle(annotatedFrame, left, right, color=(65, 189, 214),
                          thickness=3)
            cv2.putText(annotatedFrame, name, (bb.left(), bb.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                        color=(65, 189, 214), thickness=2)
        # cv2.imwrite('1.png', annotatedFrame)
        # resized = cv2.resize(annotatedFrame, (460, 360), interpolation=cv2.INTER_AREA)
        # cv2.imwrite('2.png', resized)
        plt.figure()
        plt.imshow(annotatedFrame)
        plt.xticks([])
        plt.yticks([])

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "PREVIEW",
            "data": content
        }
        plt.close()
        self.write_message(json.dumps(msg))


def main():
    # tornado.options.parse_command_line()
    # ssl_options={
    #    "certfile": "key/bestwise.crt",
    #    "keyfile": "key/bestwise.rsa",
    # }
    # http_server = tornado.httpserver.HTTPServer(Application, ssl_options)

    # http_server.listen(8080)  # xheaders=True为了可以获取到header信息
    # print('listening on %s ...' % 8080)

    # tornado.ioloop.IOLoop.instance().start()
    tornado.options.parse_command_line()
    http_server = Application()

    http_server.listen(8080, xheaders=True)  # xheaders=True为了可以获取到header信息
    print('listening on %s ...' % 8080)

    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
    # print "new server"
    # import sys
    # from twisted.python import log
    # from twisted.internet import reactor
    # log.startLogging(sys.stdout)
    # factory = WebSocketServerFactory(u"ws://0.0.0.0:9000")
    # factory.protocol = MyServerProtocol
    # reactor.listenTCP(9000, factory)
    # reactor.run()