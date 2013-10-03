clear all
close all
importfile('performance_sweep_island_long.csv');

figure
[AX,H1,H2] = plotyy(TraceDepth, RunTimeMS_SC, TraceDepth, NumBounces);
hold all
plot(TraceDepth, RunTimeMS_NoSC, 'r')

ylabel('')
xlabel('Max Trace Depth');
set(get(AX(1),'Ylabel'),'String','Time Per Frame (ms)') 
set(get(AX(2),'Ylabel'),'String','Average Bounces Per Ray') 

legend('Stream Compaction Runtime','No Stream Compaction Runtime', 'Avg Num Bounces');

title('Open Environment (Sundial)')