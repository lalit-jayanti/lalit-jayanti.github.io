---
layout: page
title: Intelligent Ground Vehicle Competition (IGVC)
description: Built a vision-based autonomous differential drive robot 
img: assets/img/abhiyaan/igvc_vikram.png
importance: 1
category: work
related_publications: true
---

At [Team Abhiyaan](https://cfi.iitm.ac.in/competition-teams/team-abhiyaan), I built a vision-based autonomous differential drive robot "Vikram". With Vikram, we competed in the AutoNav and Cyber Challenges at the 30th Annual [Intelligent Ground Vehicle Competition (IGVC)](http://www.igvc.org/index.htm) in the USA.
We won 1st place in both the AutoNav Design and Cyber challenges.

In this project, I led the software team in the development of the robot. My key contributions include:
- Navigation Stack Development: Designed and implemented a robust ROS-based navigation stack on Raspberry Pi 4 and Jetson Nano, fusing sensor data from encoders, GPS and visual odometry for precise perception and navigation.
- Control and Path-Planning: Developed and tuned PID controllers, implemented path-planning algorithms and created a GPS-based global planner with a real-time local planner for efficient motion control.
- Simulation and Validation: Built a simulation using ROS and Gazebo to test and validate path-planning and perception algorithms.

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/abhiyaan/igvc_team.png" title="The team" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/abhiyaan/igvc_vikram.png" title="The robot" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The team and the robot "Vikram".
</div>

[Technical project report](lalit-jayanti/assets/pdf/t1_vikram_technical_report.pdf)

[Simulation code](https://github.com/lalit-jayanti/virat-simulation)
