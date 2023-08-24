import math

import cv2
import numpy as np

class Line:
    def __init__(self,x0,y0,x1,y1,color):
        self.weizhi=np.array([x0,y0,x1,y1])
        self.color=color

    def draw(self,img):

        cv2.line(img, (self.weizhi[0],self.weizhi[1]), (self.weizhi[2],self.weizhi[3]), self.color,2)
        return

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

        self.points=np.array(self.points)
        self.Texture=np.array(self.Texture)
        self.Vertex=np.array(self.Vertex)
        self.f=np.array(self.f)

        self.width = int(self.img.shape[1])
        self.height = int(self.img.shape[0])

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
                all_5_3=np.array([xs,ys,zs,uvxs,uvys])
                if draw_line:
                    for j in range(3):
                        x1 = xs[j]
                        x2 = xs[(j + 1) % 3]
                        y1 = ys[j]
                        y2 = ys[(j + 1) % 3]
                        cv2.line(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255,255))

                #计算光照法向量
                ve_light=np.array([self.points[x[0] - 1] for x in self.f[i]])
                ve_light_02,ve_light_01=ve_light[2]-ve_light[0],ve_light[1]-ve_light[0]
                normal_vector=np.cross(ve_light_02,ve_light_01)

                normal_vector=normal_vector/np.sqrt(np.sum(normal_vector*normal_vector))

                intensity=normal_vector@self.light_dir
                color=(255*intensity,255*intensity,255*intensity,255)
                if intensity>0:#至关重要
                    if by_dot:
                        self.draw_shader_by_buffer(xs,ys,zs,color)
                    else:
                        self.draw_shader_byline(all_5_3,intensity)

        else:print("error in draw_line")

    def draw_shader_byline(self,all_5_3,intensity):#all_5_3 0:xs,1:ys,2:zs,3:uvxs,4:uvys
        #排序
        if (all_5_3[1][0]== all_5_3[1][1] and all_5_3[1][1] == all_5_3[1][2]):
            return
        if all_5_3[1][0] >all_5_3[1][1]:
            all_5_3=all_5_3[:,[1,0,2]]
        if all_5_3[1][1] > all_5_3[1][2]:
            all_5_3=all_5_3[:,[0,2,1]]
        if all_5_3[1][0] > all_5_3[1][1]:
            all_5_3=all_5_3[:,[1,0,2]]

        total_height=int(all_5_3[1][2]-all_5_3[1][0])
        for i in range(total_height):
            second_half = (i > (all_5_3[1][1]-all_5_3[1][0]) or (all_5_3[1][1]==all_5_3[1][0]))
            segment_height =  (all_5_3[1][2]-all_5_3[1][1]) if second_half else all_5_3[1][1]-all_5_3[1][0]
            alpha = float(i) / total_height
            beta = float((i-((all_5_3[1][1]-all_5_3[1][0]) if second_half else 0))/segment_height)
            a=int(all_5_3[0][0] + (all_5_3[0][2]-all_5_3[0][0])*alpha)

            if second_half :
                b = int(all_5_3[0][1] + (all_5_3[0][2] - all_5_3[0][1]) * beta)

            else:
                b= int(all_5_3[0][0] + (all_5_3[0][1] -all_5_3[0][0] )*beta)

            cv2.line(self.img, (int(a), int(all_5_3[1][0]+i)), (int(b), int(all_5_3[1][0]+i)), (255*intensity,255*intensity,255*intensity))

    def get_barycentric(self,A,B,C,P):
        ve_x = np.array([B[0]-A[0],C[0]-A[0],A[0]-P[0]])#AB,AC,PA
        ve_y=np.array([B[1]-A[1],C[1]-A[1],A[1]-P[1]])
        u_ve=np.cross(ve_x,ve_y)
        if (abs(u_ve[2]) > 1e-4):#当质心在内部，下面三个值全大于0
            return np.array([1-(u_ve[0]+u_ve[1])/u_ve[2],u_ve[0]/u_ve[2],u_ve[1]/u_ve[2]]);
        return np.array([-1, 1, 1])#当质心在边上，会为0，算在外面吧


    def draw_shader_by_buffer(self,xs,ys,zs,color):
        x_min, x_max=min(xs),max(xs)
        y_min, y_max=min(ys),max(ys)
        for i in range(x_min,x_max):
            for j in range(y_min,y_max):
                vec_zhi=self.get_barycentric([xs[0],ys[0]],[xs[1],ys[1]],[xs[2],ys[2]],np.array([i,j]))#计算质心坐标
                if (vec_zhi[0] <0 or vec_zhi[1] <0) or vec_zhi[2] <0: continue
                pz=0
                for z in range(3):
                    pz+=zs[z]*vec_zhi[z]
                if self.buffer[j][i]<pz:
                    self.buffer[j][i]=pz
                    #self.img[j,i]=np.array(color)
                    cv2.line(self.img,(i,j),(i,j),color)


    def show_img(self):
        self.img=cv2.flip(self.img,-1)

        cv2.imshow("666", self.img)
        cv2.waitKey(0)


if __name__=="__main__":
    path=r".\african_head.obj"

    face=obj_reader(path)
    face.trans()
    face.draw_all(by_dot=True)
    face.show_img()

