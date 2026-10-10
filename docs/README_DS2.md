# Data_DS2_raspberry
Repository with pcap files from the Raspberry Pi. (Version 2)

# Organization
* Alter_length - Data from test captures varying the frame size in the transmission packet.
* Data - Contains the data from the volunteer captures on the Raspberry Pi.

# Integrity

The file **valid.txt** contains, on each line, the participant number with the collections that are intact (o – ok) and those that were not collected (m – missing).

# File name structure

The files follow the naming pattern:
pre_time_bw_80_ch_36.pcap
Where:

- **pre** — prefix indicating the capture position  
- **time** — timestamp marking the start of the capture  
- **bw_80_ch_36** — network configuration used during the capture

### Example
0_2023_11_08_-_17_26_26_bw_80_ch_36.pcap

### Breakdown

| Component                | Meaning                                        |
|-------------------------|------------------------------------------------|
| `0`                     | First capture (empty room)                     |
| `2023_11_08_-_17_26_26` | Capture start time                             |
| `bw_80_ch_36`           | Network configuration (Wi-Fi bandwidth/channel) |
