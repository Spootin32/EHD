# EHD
Goal of the project is to build to first drone with EHD thruster to understand the feasibility of future EHD propelled aircraft.
Current high level plan is the following:
1. **[DONE]** Create a model EHD thruster (1D only)
2. **[DONE]** Derive the mass of the drone, the min thrust allowed, the supply voltage and power

| Parameter                       | Value     |
| ------------------------------- | --------- |
| Electrode distance              | 10.0 mm   |
| Electrode gap                   | 4.0 mm    |
| Airfoil cord                    | 7.0 mm    |
| Exit/Intake area ratio          | 2.5       |
| Number of stages                | 5         |
| Voltage breakdown               | 28 kV     |
| Voltage range needed            | 15-30 kV  |
| Expected thurst density at 25kV | 694 N/m2  |
| Expected max mass density       | 0.83 kg/L |

3. Draw a first quick design of a dual stage to assess minimum distances required between electrodes and general DOE
5. Build the first version of the thruster based on learnings from 5.
6. Design test setup to measure thrust
7. Iterate design and repeat 3.4.5.
8. Once thruster is defined, design drone around it.
9. Build drone, test it.
10. Define controls.
11. Controls calibration through test.
12. Final product release
