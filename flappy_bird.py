'''
Importing necessary libraries
'''
import pygame
import time
import os
import random
import sys
import numpy as np
import pandas as pd
import pickle
pygame.init()
pygame.font.init()
clock = pygame.time.Clock()


'''
Constant definitions and initialisation
'''

FPS=60

WIN_WIDTH = 500
WIN_HEIGHT = 800
PIPE_GAP = 130
IMPULSE_VEL=-10

INPUT_NODES = 4
HIDDEN_NODES = 12
OUTPUT_NODES = 1

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9


PURGE_SCORE = 100
GEN_LIMIT = 50

CYCLE_COUNT = 0
CYCLE_LIM = 1

POP_SIZE = 10
HIT_PENALTY  = 100000
CROSS_REWARD = 500

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bird1.png'))),pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bird2.png'))),pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','bird3.png')))]

BG_IMG = pygame.transform.scale(pygame.image.load(os.path.join('imgs','bg.png')),(WIN_WIDTH,WIN_HEIGHT))


def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

'''
Game Development Begins
'''

'''
Class Bird - Definition and Functions
'''
class Bird :
    IMGS = BIRD_IMGS
    IMG_HEIGHT = pygame.Surface.get_height(IMGS[0])
    IMG_WIDTH = pygame.Surface.get_width(IMGS[0])

    def __init__(self,x,y):
        self.x = x
        self.y = y 
        self.vel = 0
        self.present_vel=0
        self.height = y
        self.animation_duration = 2
        self.animation_count = 0
        self.img_count = 0
        self.img = self.IMGS[1]
        self.tick = 0
        self.max_rot = 25
        self.rot_vel = 10
        self.angle=0
        self.rotated_image=self.img
        self.score = 0
        self.fitness = 0
        self.weights = self.init_random_weights()
        self.alive=True


    def draw(self,win):
        self.move()
        if self.y<self.height and self.vel<0:
            self.angle=25
        elif self.y>self.height:
            self.angle-=self.rot_vel
            self.angle = max(self.angle,-90)
        rot_img = self.rot_center(self.img, self.angle)
        self.rotated_image=rot_img
        if CYCLE_COUNT % CYCLE_LIM==0:
            win.blit(rot_img,(self.x,self.y))
            text = font.render("Score - " + str(self.score),1,(255,255,255))
            win.blit(text,(WIN_WIDTH-200,20))

    def init_random_weights(self):
        weights =[np.random.normal(size=(INPUT_NODES,HIDDEN_NODES)),np.random.normal(size=HIDDEN_NODES)]
        return weights
    
    def jump(self):
        self.vel = IMPULSE_VEL
        self.height=self.y
        self.tick=0

    def move(self):
        self.tick+=1
        d = self.vel * self.tick + 0.5*self.tick**2
        if d>0 :
            d = max(16,d)
        self.y = self.height+d

        self.img = self.IMGS[((self.tick)//self.animation_duration)%3]
 
    def rot_center(self, image, angle):
        """rotate a Surface, maintaining position."""

        loc = image.get_rect().center  #rot_image is not defined 
        rot_sprite = pygame.transform.rotate(image, angle)
        rot_sprite.get_rect().center = loc
        return rot_sprite

    def get_mask(self):
        return pygame.mask.from_surface(self.rotated_image)
    
    def neuralOutput(self, params):
        hidden = sigmoid (np.dot (params , self.weights[0]))
        out = sigmoid (np.dot(hidden , self.weights[1]) )
        if out>0.5:
            return True 
        return False

'''
Class Base - Definition and Functions
'''
class Base :
    IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','base.png')))

    def __init__(self):
        self.width = pygame.Surface.get_width(self.IMG)
        self.x1 = 0
        self.x2 = self.width
        self.y=700
        self.vel=5
    
    def move(self):
        self.x1-=self.vel
        self.x2-=self.vel

        if(self.x1+self.width)<0:
            self.x1=self.x2+self.width
            temp=self.x2
            self.x2=self.x1
            self.x1=temp

    def draw(self,win):
        self.move()
        if CYCLE_COUNT % CYCLE_LIM==0:
            win.blit(self.IMG,(self.x1,self.y))
            win.blit(self.IMG,(self.x2,self.y))
'''
Class Pipe - Definition and Functions
'''       
class Pipe:
    IMG_BOTTOM = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','pipe.png')))
    IMG_TOP = pygame.transform.flip(pygame.transform.scale2x(pygame.image.load(os.path.join('imgs','pipe.png'))),False,True)
    IMG_HEIGHT = pygame.Surface.get_height(IMG_BOTTOM)
    IMG_WIDTH = pygame.Surface.get_width(IMG_BOTTOM)
    

    def __init__(self):
        self.x=WIN_WIDTH
        self.y_bottom=0
        self.y_top=0
        self.gap = PIPE_GAP
        self.initcords()
        self.vel=5
        self.passed=False
        self.toppipe_mask = pygame.mask.from_surface(self.IMG_TOP)
        self.bottompipe_mask = pygame.mask.from_surface(self.IMG_BOTTOM)


    def initcords(self):
        self.y_bottom=random.randrange(200,WIN_WIDTH+self.gap)
        self.y_top=self.y_bottom-self.gap-self.IMG_HEIGHT
    
    def move(self):
        self.x-=self.vel

    def draw(self,win):
        self.move()
        if CYCLE_COUNT % CYCLE_LIM==0:
            win.blit(self.IMG_BOTTOM,(self.x,self.y_bottom))
            win.blit(self.IMG_TOP,(self.x,self.y_top))
    
    def collide(self,bird):
        bird_mask = bird.get_mask()
        toppipe_mask = self.toppipe_mask
        bottompipe_mask = self.bottompipe_mask

        offset_top = (int(self.x-bird.x),int(self.y_top) - int(bird.y))
        offset_bottom = (int(self.x-bird.x),int(self.y_bottom) - int(bird.y))

        col_top = bird_mask.overlap(toppipe_mask,offset_top)
        col_bottom = bird_mask.overlap(bottompipe_mask,offset_bottom)

        if col_top or col_bottom:
            return True

        return False

'''
Game Development ENDS
Genetic Algorithm Begins
'''
class generation_stat :
    def __init__(self,gen_number):
        self.best_score = 0
        # self.best_NN = []
        self.generation_number=gen_number


def crossover(birds_weights):
    evolved_birds=[]
    num = len(birds_weights)
    for i in range(0,num):
        for j in range (i+1,num):
            evolved_set = crossover_helper(birds_weights[i],birds_weights[j])
            evolved_birds.append(evolved_set[0])
            evolved_birds.append(evolved_set[1])
    return evolved_birds

def convert_to_list(weights):
    arr1 = []
    for i in range(INPUT_NODES):
        for j in range (HIDDEN_NODES):
            arr1.append(weights[0][i][j])

    for i in range (HIDDEN_NODES):
        arr1.append(weights[1][i])
    
    return arr1

def convert_to_NN(arr):
    newgen_weights =[np.random.normal(size=(INPUT_NODES,HIDDEN_NODES)),np.random.normal(size=HIDDEN_NODES)]

    k = 0 
    for i in range(INPUT_NODES):
        for j in range(HIDDEN_NODES):
            newgen_weights[0][i][j] = arr[k]
            k+=1
    
    for i in range(HIDDEN_NODES):
        newgen_weights[1][i] = arr[k]
        k+=1
    return newgen_weights


    

def crossover_helper(weights1  , weights2):

    weights1 = convert_to_list(weights1)
    weights2 = convert_to_list(weights2)
    
    mean_list = []
    result_arr1 = []
    result_arr2 = []

    for i in range(len(weights1)):
        mean_list.append((weights1[i]+weights2[i])/2)

    for i in range(len(weights1)):
        ratio1 = weights1[i]/mean_list[i]
        ratio2 = weights2[i]/mean_list[i] 
        if ratio1>ratio2:
            result_arr1.append(weights1[i])
            result_arr2.append(mean_list[i])
        else: 
            result_arr1.append(weights2[i])
            result_arr2.append(mean_list[i])
       
    
    newgen_weights1 =convert_to_NN(result_arr1)
    newgen_weights2 =convert_to_NN(result_arr2)

    return [newgen_weights1,newgen_weights2]
    
def mutate(weights):
    for weight in weights:
        for i in range(INPUT_NODES):
            for j in range (HIDDEN_NODES):
                test = np.random.normal(0,1)
                if test<MUTATION_RATE:
                    weight[0][i][j]+=np.random.normal(-0.1,0.1)

        for i in range (HIDDEN_NODES):
            test = np.random.normal(0,1)
            if test<MUTATION_RATE:
                    weight[1][i]+=np.random.normal(-0.1,0.1)
    return weights

def evolve (birds):

    #SELECTION
    # list of top neural nets
    birds.sort(key=lambda x: x.fitness, reverse=True)
    top_birds = birds[:10]
    elite = []
    for bird in birds : 
        if bird.score==PURGE_SCORE:
            elite.append(bird)
    if len(elite)<10:
        elite=top_birds
    top_birds_weights = []
    for bird in top_birds:
        top_birds_weights.append(bird.weights)
    print(birds[0].fitness)

    
    #CROSSOVER
    evolved_weights = crossover(top_birds_weights)
       

    #MUTATION
    evolved_weights = mutate(evolved_weights)
    evolved_weights_fin = []
    for bird in elite:
        evolved_weights_fin.append(bird.weights)
    for i in range(0,min(len(evolved_weights),100-len(evolved_weights_fin))):
        evolved_weights_fin.append(evolved_weights[i]) 

    #New list of weights
    evolved_birds = []
    for idx in range (0,100):
        evolved_bird = Bird(100,100)
        evolved_bird.weights = evolved_weights_fin[idx]
        evolved_birds.append(evolved_bird)
    return evolved_birds

def store_stats(birds,df,generation):
    birds.sort(key=lambda x: x.score, reverse=True)
    max_score = birds[0].score
    avg = 0
    for i in range (0,len(birds)):
        avg+=birds[i].score
    avg/=len(birds)
    df["Gen "+ str(generation)] = avg
    return df

def high_score(birds):
    birds.sort(key=lambda x: x.score, reverse=True)
    avg = 0
    for i in range (0,len(birds)):
        avg+=birds[i].score
    avg/=len(birds)
    return birds[0].score,avg

'''
Genetic Aglorithm ENDS
Pygame drawing functions Begin
'''
def draw_lines_vis(win,birds,param_x,param_y):
    for bird in birds:
        cen_x = bird.x+bird.IMG_WIDTH//2
        cen_y = bird.y + bird.IMG_HEIGHT//2
        up_y = param_y-PIPE_GAP//2
        down_y = param_y+PIPE_GAP//2
        pygame.draw.lines(win,(0,0,0),False,[(cen_x,cen_y),(param_x,up_y)],2)
        pygame.draw.lines(win,(255,255,255),False,[(cen_x,cen_y),(param_x,down_y)],2)

def win_draw(win,birds,base,pipes,generation):
    if CYCLE_COUNT % CYCLE_LIM==0:
        win.blit(BG_IMG,(0,0)) #Background Image
    for pipe in pipes :
        pipe.draw(win)
    base.draw(win)
    if CYCLE_COUNT % CYCLE_LIM==0:
        draw_lines_vis(win,birds,param_pipe_x,param_pipe_y)
    for bird in birds:
        bird.draw(win) #Drawing Bird

    if CYCLE_COUNT % CYCLE_LIM==0:
        pygame.draw.circle(win, (255,0,0), (param_pipe_x, param_pipe_y), 3)

        # draw_lines_vis(win,birds,param_pipe_x,param_pipe_y)

        gen_text = font.render("Generation - " + str(generation),1,(255,255,255))
        win.blit(gen_text,(30,20))

        mut_text = smallFont.render("Mutation Rate - " + str(MUTATION_RATE),1,(0,0,0))
        win.blit(mut_text,(30,45))

        cross_text = smallFont.render("Crossover Rate - " + str(CROSSOVER_RATE),1,(0,0,0))
        win.blit(cross_text,(30,60))

        popsize_text = smallFont.render("Population Size - " + str(POP_SIZE),1,(0,0,0))
        win.blit(popsize_text,(30,75))

        pipegap_text = smallFont.render("Pipe Gap - " + str(PIPE_GAP),1,(0,0,0))
        win.blit(pipegap_text,(30,90))

        alive_text = smallFont.render("Birds Alive - " + str(len(birds)),1,(0,0,0))
        win.blit(alive_text,(30,105))
        pygame.display.update() #Updating screen


'''
Function to RUN_GAME
'''
def runGame(birds,generation):
    
    #Initalising window 
    run=True

    #Initilaising moving floor
    base = Base()

    
    #pipestuff
    pipeTimer = 0
    pipeTimerLim = 90
    pipes = []

    #Return a list of birds with updated scored
    updated_birds = []

    #Main Game Loop
    while run:
        global CYCLE_COUNT
        global CYCLE_LIM
        #Frame Rate 60FPS
        global FPS
        clock.tick(FPS)
        # print("FPS: {}".format(clock.get_fps()))
        
        #Game Logic
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run=False                
                pygame.quit()
        
        keys = pygame.key.get_pressed()

        if keys[pygame.K_1]:
            CYCLE_LIM=1
        if keys[pygame.K_2]:
            CYCLE_LIM=10
        if keys[pygame.K_3]:
            CYCLE_LIM=50
        if keys[pygame.K_4]:
            CYCLE_LIM=100000  
        if keys[pygame.K_5]:
            FPS=0
        if keys[pygame.K_6]:
            FPS=1
        if keys[pygame.K_7]:
            FPS=10000
        
        #Adding and removing pipes the pipes on screen
        if(pipeTimer==0):
            pipes.append(Pipe())
        pipeTimer=(pipeTimer+1)%pipeTimerLim

        for pipe in pipes :
            if pipe.x + pipe.IMG_WIDTH<0:
                pipes.remove(pipe)

        #Bird interacts with NN
        
        nextpipe = pipes[0] if pipes[0].x + pipes[0].IMG_WIDTH - birds[0].x>0 else pipes[1]

        
        global param_pipe_x 
        param_pipe_x = nextpipe.x+ nextpipe.IMG_WIDTH//2
        global param_pipe_y
        param_pipe_y = nextpipe.y_top + nextpipe.IMG_HEIGHT + nextpipe.gap//2
        global draw_help_x
        draw_help_x = nextpipe.x + nextpipe.IMG_WIDTH//2
        d=birds[0]
        d_cen_x=d.x+d.IMG_WIDTH//2
        if d_cen_x<nextpipe.x :
            param_pipe_x = nextpipe.x
            param_pipe_y = nextpipe.y_top + nextpipe.IMG_HEIGHT + nextpipe.gap//2
        elif d_cen_x>nextpipe.x and d_cen_x<nextpipe.x+nextpipe.IMG_WIDTH//2:
            param_pipe_x = nextpipe.x + nextpipe.IMG_WIDTH//2
            param_pipe_y = nextpipe.y_top + nextpipe.IMG_HEIGHT + nextpipe.gap//2    
        else :
            param_pipe_x = nextpipe.x + nextpipe.IMG_WIDTH
            param_pipe_y = nextpipe.y_top + nextpipe.IMG_HEIGHT + nextpipe.gap//2    

        for bird in birds:
            if(bird.alive==True):
                bird.fitness+=int(100/(1+abs(bird.y + bird.IMG_HEIGHT//2 -param_pipe_y)))
                # print("Loc1 " + str(int(1000/(1+abs(bird.y-param_pipe_y)))))
                params = [bird.tick,param_pipe_x-bird.x, bird.y - (nextpipe.y_top + nextpipe.IMG_HEIGHT) ,nextpipe.y_bottom-(bird.y)]
                if bird.neuralOutput(params):
                    bird.jump()
                
        
        for pipe in pipes:
            for bird in birds:
                if pipe.collide(bird) or bird.score>=PURGE_SCORE:
                    if bird.score==100:
                        print(bird.weights)
                    bird.alive=False
                    updated_birds.append(bird)
                    birds.remove(bird)
                    # print("Collision detected")
                elif bird.y>=700 or bird.y<0 :
                    bird.alive=False
                    bird.fitness-=HIT_PENALTY
                    updated_birds.append(bird)
                    birds.remove(bird)

            if len(birds)>0 and pipe.x+pipe.IMG_WIDTH<birds[0].x and pipe.passed==False:
                for bird in birds :
                    bird.score+=1
                    bird.fitness+=CROSS_REWARD
                pipe.passed=True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            for bird in birds:
                bird.alive=False
                updated_birds.append(bird)
                birds.remove(bird)
        
        if(len(birds)==0):
            return updated_birds
        
        
        
        win_draw(win,birds,base,pipes,generation)
        CYCLE_COUNT= CYCLE_COUNT + 1

'''
Main() Function
'''

def main():

    global win
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))

    global largeFont, smallFont, font
    font = pygame.font.SysFont('helvetica',30)
    largeFont = pygame.font.SysFont('helvetica', 80)
    smallFont = pygame.font.SysFont('helvetica', 20)

    birds = []
    for i in range (0,POP_SIZE):
        birds.append(Bird(100,100))
    
    #DF for this series of population statistics is df_gen
    dict_run ={}
    dict_run['Mutation Rate'] = MUTATION_RATE

    for generation in range (0,GEN_LIMIT):
        generation_end = runGame(birds,generation)

        max_score,avg_score =high_score(generation_end)
        print("The generation is " + str(generation) + " MUT " +str(MUTATION_RATE) + " MAX SCORE " + str(max_score) + " AVG SCORE "+str(avg_score))

        dict_run = store_stats(generation_end,dict_run,generation)
        birds = evolve(generation_end)

    

main()

pygame.quit()