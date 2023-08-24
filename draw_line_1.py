import math

import cv2
import numpy as np

class obj_reader:
    def __init__(self,path):
        self.points = []
        # v
        # 几何体顶点
        self.Texture=[]
        # vt
        # 贴图坐标点
        self.Vertex=[]
        # vn
        # 顶点法线
        self.Parameter=[]
        # vp
        # 参数空格顶点
        self.f=[]
        self.img = np.zeros((800, 800, 3), np.uint8)

        with open(path) as file:
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")

                if strs[0] == "v":
                    self.points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                elif strs[0] == "vt":
                    self.Texture.append((float(strs[2]), float(strs[3]), float(strs[4])))
                elif strs[0] == "vn":
                    self.Vertex.append((float(strs[2]), float(strs[3]), float(strs[4])))
                elif strs[0] == "f":
                    self.f.append([[int(data) for data in x.split('/')] for x in strs[1:]])
                else:
                    continue
        file.close()

        self.wenli=cv2.imread('./african_head_diffuse.jpg')#纹理读取
        self.points=np.array(self.points)
        self.Texture=np.array(self.Texture)
        self.Vertex=np.array(self.Vertex)
        self.f=np.array(self.f)

        self.width = int(self.img.shape[1])
        self.height = int(self.img.shape[0])
        self.light_dir = np.array([0, 0, -1])  # 光照方向
        self.buffer = np.ones((self.img.shape[0], self.img.shape[1])) * -1e6  # 设置成一个比较大的值

    def trans(self):
        #c=
        self.toushi=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.light_dir=np.array([0,0,-1])#光照方向
        #x,y,z=math.pi/24,math.pi/3,math.pi/12#上下，侧面，歪头
        x,y,z=0,0,0
        self.c2wx=np.array([[1,0,0,0],
                           [0,math.cos(x),-math.sin(x),0],
                           [0,math.sin(x),math.cos(x),0],
                           [0,0,0,1]])
        self.c2wy = np.array([[math.cos(y), 0, math.sin(y),0 ],
                              [0,1,0,0],
                              [-math.sin(y),math.cos(y),1,0],
                              [0,0,0,1]])
        self.c2wz = np.array([[math.cos(z),-math.sin(z), 0, 0],
                              [math.sin(z),math.cos(z),0,0],
                              [0,0,1,0],
                              [0,0,0,1]])
        self.c2t = np.array([[1,0, 0, 0],
                             [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.points2=[]
        self.c2w=self.c2wz@self.c2wy@self.c2wx@self.c2t
        self.c2w_toushi=self.toushi@self.c2w
        for point in self.points:
            new_point=self.c2w_toushi @ np.append(point, 1)
            self.points2.append([new_point[0]/new_point[3],new_point[1]/new_point[3],new_point[2]/new_point[3]])
        self.points=np.array(self.points2)

    def draw_all(self,draw_line=False,by_dot=False):

        if self.f.shape[0] > 1:
            for i in range(self.f.shape[0]):
                print(i)
                xs=[int((self.points[x[0] - 1][0] + 1) * self.width / 2) for x in self.f[i]]
                ys=[int((self.points[x[0] - 1][1] + 1) * self.height / 2) for x in self.f[i]]
                zs = [int((self.points[x[0] - 1][2] + 1)  / 2) for x in self.f[i]]

                uvxs = [int((self.Texture[x[1] - 1][0] ) * 1024 ) for x in self.f[i]]
                uvys = [int((self.Texture[x[1] - 1][1] ) * 1024 ) for x in self.f[i]]

                for j in range(3):
                    x1 = xs[j]
                    x2 = xs[(j + 1) % 3]
                    y1 = ys[j]
                    y2 = ys[(j + 1) % 3]
                    cv2.line(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255,255))


        else:print("error in draw_line")

    def show_img(self):
        self.img=cv2.flip(self.img,-1)

        cv2.imshow("666", self.img)
        cv2.waitKey(0)


if __name__=="__main__":
    path=r".\african_head.obj"

    face=obj_reader(path)
    face.trans()
    face.draw_all()
    face.show_img()

