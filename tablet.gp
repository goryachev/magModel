set style data p
set parametric
plot "tab4.dat" u 2:3, 3*sin(t),3*cos(t)
pause -1
plot 19.5*sin(t),19.5*cos(t), 14*sin(t),14*cos(t), 10*sin(t),10*cos(t), "tab20.dat" u 2:3 lt 1, "tab14.dat" u 2:3 lt 2, "tab11.dat" u 2:3 lt 3
pause -1
plot 10*sin(t),10*cos(t), "tab10.dat" u 2:3 lt 1, "tab11.dat" u 2:3 lt 2
pause -1
plot 10*sin(t),10*cos(t), 14*sin(t),14*cos(t), 19.5*sin(t),19.5*cos(t), "tab19.dat" u 2:3 lt 3, "tab20.dat" u 2:3 lt 4
pause -1
plot "tab11.dat" u 2:3, 10*sin(t),10*cos(t)
pause -1
plot "tab14.dat" u 2:3, 14*sin(t),14*cos(t)
pause -1
plot "tab20.dat" u 2:3, 19.5*sin(t),19.5*cos(t)
pause -1
