# 유전 알고리즘을 이용한 최속강하곡선 그리기
# 코드의 작성방식은 PEP8 (Program Enhance Proposal 8)에 의거합니다.

import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from random import uniform, randint
from math import sqrt, cos, sin, pi, ceil

r = 20#100                         # 맵 크기
x_len = 2*r                     # 그래프상 x범위
x_size = 20#100                    # 구간 개수(염기서열 길이)
y_len = 2*r                     # 그래프상 y범위
y_size = 2*r                    # 구간 개수

x = []
for i in range(x_size + 1):
    x.append(i*pi*r/x_size)

G = 9.80665                     # 중력가속도
T = sqrt(r / G)*pi              # 최속강하시간
interval_delay = 10              # 단위: ms

# 정규화 함수: 배열의 msum과 ysize가 같도록 만듬
def generalization(m_arr0):
    # m_arr0 : 기울기 배열 입력
    m_arr = copy.deepcopy(m_arr0)   # 깊은 복사, 이래야 원래 값에 영향 x
    msum = 0                    # 기울기 합
    for i in m_arr:
        msum += i

    delta = y_len / msum * -1

    for i in range(len(m_arr)): 
        m_arr[i] *= delta
    return m_arr

# 시간 합 함수: 시간 합을 구함
def sum_time(m_arr):
    # m_arr : 기울기 배열 입력
    sum = 0
    v0 = 0
    for m in m_arr:
        M = abs(m)
        batch = r * pi / x_size
        h = M * batch
        # feat. 가속도 정의, m <= 0이라는 가정 하에 만듦
        # a = h * G / sqrt(batch**2 + h**2)
        a = G * M / sqrt(1+M**2)
        # 나중속도 산출
        if M == 0:
            after_v = v0
        else:
            # feat. 역학적E보존
            after_v = sqrt(2*G*h + v0**2)
        # feat. 가속도 정의 이용, t 산출
        if a == 0:
            if v0 == 0:
                return 1000
            else :
                t = pi/v0
        else:
            t = (after_v - v0)/a
        sum += t
        v0 = after_v
    return sum

# 적합도 함수: 적합도 계산
def fitness(t):
    # t : sum_time입력받음
    f = T/t*100
    return f

# 사이클로이드 그림 함수: 사이클로이드를 그림
def draw_cycloid(r):
  # r : 맵 크기 입력
  x = []                        # x좌표 리스트 만듦
  y = []                        # y좌표 리스트 만듦

  for theta in np.linspace(0, np.pi, 100): # theta변수를 0 에서 π 까지 반복함
    x.append(r*(theta - sin(theta))) # x 리스트에 매개변수함수값을 추가시킴
    y.append(10 -(r*(1 - cos(theta)))) # y 리스트에 매개변수함수값을 추가시킴

  plt.figure(figsize=(pi*2, 4))# 그래프 비율 조정
  plt.plot(x,y)                 # matplotlib.piplot을 이용해 그래프 그리기
  plt.title('Cycloid')
  plt.xlabel('x')
  plt.ylabel('y')
  #plt.xlim([0, r*pi])
  #plt.ylim([0, 2*r])
  plt.show()                    # 그래프 출력하기

# 유전 최속강하곡선 그림 함수: 유전 알고리즘으로 만들어진 사이클로이드를 그림
def draw_GA_Brachistochrone(m_arr, generation, fitness):
    # m_arr : 기울기 배열 입력
    # generation : 세대 입력
    # fitness : 적합도 입력
    y = [y_len]
    for i in range(x_size):
        y.append(y[-1] + m_arr[i])
    
    plt.plot(x, y)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title(f'GA_Brachistochrone_{generation}')
    plt.xlim([0, r*pi])
    plt.ylim([0, 2*r])
    plt.text(0.5, 0.05, f'Generation: {generation}\nFitness: {fitness:.2f}', 
             transform=plt.gca().transAxes, fontsize=10, ha='center')
    plt.show()

# 유전 함수: 유전하는 함수
def selection(num, min_m, max_gen, mut_chance, mut_rate, rate_arr):
    # 1.2. 초기화(Initialization): 0세대 생성
    parent = []                 # 부모 배열
    gen = 0                     # 세대수
    generation = []             # 개체군 (세대수) 배열
    gen_maxfitness = [0]         # 개체군 (세대 최고 적합도) 배열
    gen_avefitness = []         # 개체군 (세대 평균 적합도) 배열
    gen_minfitness = []         # 개체군 (세대 최저 적합도) 배열
    gen_maxBranchiststochrone = []            # 개체군 (세대 최고 자녀) 배열
    sum_fitness = 0             # 세대 적합도 합
    max_fitness_gen = 0               # 개체군 최대 적합도
    rate_num_arr = list(map(int, (num * (np.array(rate_arr))).tolist()))   # 비율 배열
    # parent = [[부모 1의 적합도, 부모 1의 강하시간, 부모 1의 염기서열], [부모 2의 적합도, 부모 2의 강하시간, 부모 2의 염기서열], ...]
    for i in range(num):        # 부모 배열 생성
        parent.append([0, 0, []])
        
        for j in range(x_size): # 부모 염기서열 생성
            parent[i][2].append(uniform(min_m, 0))
        
        parent[i][2] = generalization(parent[i][2])
    
    # 2.6. 종료 조건 검사(Termination Condition Check): 반복
    while gen < max_gen:
    # 2.1. 적합도 평가(Fitness Evaluation): 부모 사살
        fitness_arr = []        # 부모 적합도 배열
        time_arr = []           # 부모 강하시간 배열
        sum_fitness = 0
        for i in range(num):
            time_arr.append(sum_time(parent[i][2]))
            parent[i][1] = copy.deepcopy(time_arr[i])
            fitness_arr.append(fitness(time_arr[-1]))
            parent[i][0] = copy.deepcopy(fitness_arr[i])
            sum_fitness += fitness_arr[i]
        fitness_arr.sort(reverse=True)      # 부모 적합도 배열
        time_arr.sort()                     # 부모 강하시간 배열
        parent.sort(reverse=True)           # 먼저 적합도 정렬, 이후 강하시간 정렬, 이후 염기서열 정렬, 그러나 적합도 정렬에서 끝

        # Fitnesses of Population 그래프 그리기
        gen += 1
        generation.append(gen)
        gen_maxfitness.append(fitness_arr[0])
        gen_maxBranchiststochrone.append(parent[0])
        if gen_maxfitness[-1] > gen_maxfitness[-2]:
            max_fitness_gen = gen
        gen_avefitness.append(sum_fitness/num)
        gen_minfitness.append(fitness_arr[-1])

        """
        plt.plot(list(range(num)), fitness_arr)
        plt.title('Fitness Table')
        plt.xlim([0, num])
        plt.ylim([0, 100])
        plt.show()
        """

        # 세대 출력
        if gen % (max_gen//5) == 0:
            print('gen :', gen)
            #draw_GA_Brachistochrone(parent[0][2], gen, parent[0][0])

    # 2.2. 선택(Selection): 순위 선택
        parent = parent[:rate_num_arr[0]]
        child = []
       
    # 2.3. 교차(Crossover): 부모 조작
        # 2.3.1. 선택 교차 : 기존에 선택된 부모의 염기서열이 그대로 보존되어 교차
        parent *= ceil(1/rate_arr[0])
        child = copy.deepcopy(parent[:rate_num_arr[1]])

        # 2.3.2. 결혼 교차 : 기존에 선택된 두 부모의 염기서열이 2점 교차되어 새로운 두 자녀가 되어 교차
        for i in range(int(0.5*rate_num_arr[2])):
                # cross_arr = [2점 교차될 선택 부모 1, 2점 교차될 선택 부모 2, 2점 교차될 위치 1, 2점 교차될 위치 2]
                cross_arr = []
                for j in range(4):
                    if j < 2:
                        cross_arr.append(randint(0, rate_num_arr[0]))
                    else:
                        cross_arr.append((randint(0, x_size)))
                    
                    if j % 2 == 1:
                        while cross_arr[j-1] == cross_arr[j]:
                            if j == 1:
                                cross_arr[j] = randint(0, rate_num_arr[0])
                                cross_arr[j-1] = randint(0, rate_num_arr[0])
                            else:
                                cross_arr[j] = randint(0, x_size)
                                cross_arr[j-1] = randint(0, x_size)
                        
                        if cross_arr[j-1] > cross_arr[j]:   #오름차순 배열
                            cross_arr[j-1], cross_arr[j] = cross_arr[j], cross_arr[j-1]

                child.append(parent[cross_arr[0]])
                child.append(parent[cross_arr[1]])
                child[-1][2][cross_arr[2]:cross_arr[3]+1], child[-2][2][cross_arr[2]:cross_arr[3]+1] \
                = child[-2][2][cross_arr[2]:cross_arr[3]+1], child[-1][2][cross_arr[2]:cross_arr[3]+1]

        # 2.3.3. 생성 교차 : 기존에 선택된 부모의 염기서열과 독립적이지만 같은 분포로(iid) 생성되어 교차
        for i in range(rate_num_arr[3]):
            child.append([0, 0, []])

            for j in range(x_size): # 자식 염기서열 생성
                child[-1][2].append(uniform(min_m, 0))

    # 2.4. 돌연변이(Mutation): 부모 변이
        for i in range(num):
            for j in range(x_size):
                point = uniform(0, 1)
                if(point < mut_chance):
                    child[i][2][j] *= uniform(1-mut_rate, 1+mut_rate)

    # 2.5. 교체(Replacement): 자녀 생성
        for i in range(num):
            child[i][2] = generalization(child[i][2])
        parent=copy.deepcopy(child)
        child = []
        
# 3.1. 출력(Output): 출력
    gen_maxfitness.pop(0)

    # FP 그래프 그리기
    #plt.plot(generation, gen_maxfitness, marker = 'o')
    #plt.plot(generation, gen_avefitness, marker = 'o')
    #plt.plot(generation, gen_minfitness, marker = 'o')
    #plt.title('Fitness of Population: FP')
    #plt.legend(['max', 'ave', 'min'])
    #plt.show()

    # 적합도 출력
    print(separator_str)
    print('max fitness gen:', max_fitness_gen)
    print('max fitness:', parent[0][0], '%')
    print('GA_B time:', parent[0][1], 's')
    print('time difference between GA_B and C:', parent[0][1] - T, 's')
    print(separator_str)

    #곡선 출력
    #draw_GA_Brachistochrone(parent[0][2], gen, parent[0][0])

    # SP그래프 그리기
    # FP, BP 초기 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # 1 행 2 열의 subplot 생성
    ax1.axhline(0, color='black', linewidth=0.5)  # x 축
    ax1.axvline(0, color='black', linewidth=0.5)  # y 축
    ax1.set_xlim(0, r*pi)
    ax1.set_ylim(0, 2*r)
    line1, = ax1.plot([], [])
    text1 = ax1.text(0.5, 0.05, '', transform=ax1.transAxes, fontsize=10, ha='center')
    ax1.set_title('Brachistochrone of Population: BP')

    ax2.set_xlim(0, max_gen)
    ax2.set_ylim(0, 100)
    line2_max, = ax2.plot([], [], marker='o', label='max')
    line2_ave, = ax2.plot([], [], marker='o', label='ave')
    line2_min, = ax2.plot([], [], marker='o', label='min')
    ax2.set_title('Fitness of Population: FP')
    ax2.legend()

    max_data_x = []
    max_data_y = []
    ave_data_x = []
    ave_data_y = []
    min_data_x = []
    min_data_y = []

    def init():
        line1.set_data([], [])
        text1.set_text('')
        line2_max.set_data([], [])
        line2_ave.set_data([], [])
        line2_min.set_data([], [])
        return line1, text1, line2_max, line2_ave, line2_min

    def animate(generation):
        best_individual = gen_maxBranchiststochrone[generation][2]
        y = [y_len]
        for i in range(x_size):
            y.append(y[-1] + best_individual[i])

        line1.set_data(x, y)
        text1.set_text(f'Generation: {generation}\nFitness: {gen_maxfitness[generation]:.5f}')

        #color = plt.cm.viridis(generation / max_gen)  # 각 세대별 다른 색상
        #line1.set_color(color)

        max_data_x.append(generation)
        max_data_y.append(gen_maxfitness[generation])
        ave_data_x.append(generation)
        ave_data_y.append(gen_avefitness[generation])
        min_data_x.append(generation)
        min_data_y.append(gen_minfitness[generation])

        line2_max.set_data(max_data_x, max_data_y)
        line2_ave.set_data(ave_data_x, ave_data_y)
        line2_min.set_data(min_data_x, min_data_y)

        ax2.relim()
        ax2.autoscale_view(scaley=True)

        return line1, text1, line2_max, line2_ave, line2_min

    # 애니메이션 생성
    ani = FuncAnimation(fig, animate, frames=range(max_gen), init_func=init, blit=True, interval=interval_delay, repeat=False)
    fig.suptitle('Statistics of Population: SP')
    plt.show()


"""알고리즘 개요"""
# 1.1. 입력(Input): 입력
# 1.2. 초기화(Initialization): 0세대 생성
# 2.1. 적합도 평가(Fitness Evaluation): 부모 사살
# 2.2. 선택(Selection): 순위 선택
# 2.3. 교차(Crossover): 부모 조작
    # 2.3.1. 선택 교차: 기존에 선택된 부모의 염기서열이 그대로 보존되어 교차
    # 2.3.2. 결혼 교차: 기존에 선택된 두 부모의 염기서열이 2점 교차되어 새로운 두 자녀가 되어 교차
    # 2.3.3. 생성 교차: 기존에 선택된 부모의 염기서열과 독립적이지만 같은 분포로(iid) 생성되어 교차
# 2.4. 돌연변이(Mutation): 부모 변이
# 2.5. 교체(Replacement): 자녀 생성
# 2.6. 종료 조건 검사(Termination Condition Check): 반복
# 3.1. 출력(Output): 출력

# 1.1. 입력(Input): 입력
separator_str = '-' * 55
max_gen = 200                    # 최대 세대 수
num = 120                       # 부모 수: (0, +inf), 비율 배열과 곱했을 때 자연수(rate_arr[2]*0.5)
min_m = -10                     # 최소 기울기 값: (-inf, 0)
mut_chance = 0.3                # 돌연변이 교체 비율: [0, 1]
mut_rate = 0.3                  # 돌연변이시 바뀌 최대 비율: [0, +inf)
rate_arr = [0.1, 0.3, 0.6, 0.1] # 비율 배열: [0, 1]
# 비율 배열 = [선택 부모 비율, 교체 선택 비율, 교체 교차 비율, 교체 생성 비율]
# 교체 선택 비율+교체 교차 비율+교체 생성 비율 = 1
# 실질적 최소 기울기 값: minm*(1+mut_chance)

print(separator_str)
# 제목 출력
print('Finding Brachistochrone by Genetic Algorithm')
print('C time:', T, 's')
# 부모 정보 출력
print(separator_str)
print('total child:', num)
print('min_m:', min_m)
print('mut_chance:', mut_chance)
print('mut_rate:', mut_rate)
print('rate_arr:', rate_arr)
print('DNA sequence length: ', x_size)
print('map size:', r)
print(separator_str)

# 유전 알고리즘 시작
selection(num, min_m, max_gen, mut_chance, mut_rate, rate_arr)
#draw_cycloid(r)
