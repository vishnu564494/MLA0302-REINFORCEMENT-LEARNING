import numpy as np
energy=[45,60,55,70,80,65,50,75]
demand=[50,58,60,72,78,68,52,70]
actions=["Store","Supply","Idle"]
policy=np.random.rand(3)
policy=policy/policy.sum()
lr=0.05
episodes=300
for ep in range(episodes):
    reward_sum=0
    grad=np.zeros(3)
    for i in range(len(energy)):
        action=np.random.choice([0,1,2],p=policy)
        if action==0:
            reward=5 if energy[i]>demand[i] else -3
        elif action==1:
            reward=8-abs(energy[i]-demand[i])
        else:
            reward=2
        reward_sum+=reward
        grad[action]+=reward
    grad=grad/(np.linalg.norm(grad)+1e-6)
    policy+=lr*grad
    policy=np.maximum(policy,0.01)
    policy=policy/policy.sum()
print("SMART GRID ENERGY MANAGEMENT\n")
print("Energy Production :",energy)
print("Energy Demand     :",demand)
print("\nLearned Policy")
for i in range(3):
    print(actions[i],":",round(policy[i],3))
print("\nEnergy Decisions")
total_cost=0
balance=0
for i in range(len(energy)):
    action=np.argmax(policy)
    if action==0:
        decision="Store Energy"
        cost=2
    elif action==1:
        decision="Supply Energy"
        cost=1
    else:
        decision="Idle"
        cost=3
    total_cost+=cost
    balance+=abs(energy[i]-demand[i])
    print("Hour",i+1)
    print("Production :",energy[i],"MW")
    print("Demand     :",demand[i],"MW")
    print("Decision   :",decision)
    print("Cost       :",cost)
    print()
print("Total Cost :",total_cost)
print("Grid Imbalance :",balance,"MW")
print("Policy Optimization Completed")
