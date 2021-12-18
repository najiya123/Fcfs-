# Fcfs-

#include<stdio.h>
void main()
{
    int i,n,at[10],bt[10],ct[10],tat[10],wt[10],tot_tat,tot_wt;
    float avg_tat,avg_wt;
    //scanning the number of process
    printf("\nEnter the # of process : ");
    scanf("%d",&n);
    //Scanning the arrival time and burst time of each process
    printf("\nEnter the AT and BT of each process according to the order of their arrival respectively: ");
    for(i=0;i<n;i++)
    {
        scanf("%d%d",&at[i],&bt[i]); 
    }
    //calculating completion time
    ct[0]=at[0]+bt[0]; //initial values of ct[0]
    for(i=1;i<n;i++)
    {
        if(at[i]<=ct[i-1])
        {
            ct[i] = ct[i-1] + bt[i];
        }
        else
        {
            ct[i] = at[i]+bt[i];
        }
    }
    //calculating the turn around  time and waiting time
    for(i=0;i<n;i++)
    {
        tat[i] = ct[i]-at[i];
        wt[i] = tat[i]-bt[i];
    }
    // Printing the result
    printf("\nPID\tAT\tBT\tCT\tTAT\tWT\n");
    for(i=0;i<n;i++)
    {
        printf("\n%d\t%d\t%d\t%d\t%d\t%d",i+1,at[i],bt[i],ct[i],tat[i],wt[i]);
    }
    //Printing avg_tat and avg_wt
    tot_tat=0;
    tot_wt=0;
    for(i=0;i<n;i++)
    {
        tot_tat += tat[i];
        tot_wt += wt[i];
    }
    avg_tat = (float)tot_tat/(float)n;
    avg_wt= (float)tot_wt/(float)n;
    printf("\nAverage TAT = %.2f and Average WT = %.2f",avg_tat,avg_wt);
}
