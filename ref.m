clear all
clc
% Simulation duration
tsimul = 10;
% Wind parameters
w_speed =14;
% Nominal machine electrical parameters
Pnom = 2e6; %Nominal power mach
Vnom = 960; %Nominal voltage mach
% Turbine parameters
% Cp values
f=47;
c1 = 0.44;
c2 = 125;
c3 = 0;
c4 = 0;
c5 = 0;
c6 = 6.94;
c7 = 16.5;
c8 = 0;
c9 = -0.002;
R_turbine = 38;        % Turbine radius
wtnominal=16*2*pi/60;
A = pi*R_turbine^2;    % Area swept by the blades
rho =1.225;            % Air Density
angle_pitch=0;         % Pitch Angle
w_ini=2*pi*25/80;      %
Jtot=9e6;              % 
n_multiplier = 80;     %
% Generator Electrial Parameters
Rs = 0.005;
Xs = 2*pi*f*4e-4;
Rr = 0.009;
Xr = 2*pi*f*3E-4; 
Xm = 2*pi*f*15E-3; 
Rm = 140;
V = Vnom/sqrt(3); 
pols = 2;   
wg = 2*pi*f/pols;
wt=wg/n_multiplier;
lambda=(wt*R_turbine)/w_speed;
tsr=0:.1:17; 
t=0:.1:tsimul;
k1=(tsr+c8*angle_pitch).^(-1)-c9 /(1+angle_pitch^3);
Cp=c1*(c2*(1/lambda)-c6)*exp(-c7*(1/lambda));
Power=0.5*rho*A*Cp*w_speed^3
cp=max(0,c1*(c2*k1-c3*angle_pitch-c4*angle_pitch^c5-c6).*exp(-c7*k1)); 
%figure(); % Crea una nova f i g u r a
%plot(tsr,cp,'LineWidth',2);grid on;
%xlabel('\lambda','FontSize',14);
%ylabel('Cp','FontSize',14);     