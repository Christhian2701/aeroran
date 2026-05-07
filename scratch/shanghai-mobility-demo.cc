/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Shanghai 63 UE Mobility - Working Demo with Trace Output
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"
#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Shanghai63UEs");

int 
main (int argc, char *argv[])
{
    // Parameters
    double simTime = 30.0;
    uint32_t nUEs = 10; // Reduce for testing
    
    CommandLine cmd;
    cmd.AddValue ("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue ("nUEs", "Number of UEs", nUEs);
    cmd.Parse (argc, argv);
    
    NS_LOG_INFO ("Starting Shanghai UE Mobility Demo");
    NS_LOG_INFO ("Simulation time: " << simTime << " seconds");
    NS_LOG_INFO ("Number of UEs: " << nUEs);
    
    // Create UE nodes
    NodeContainer ueNodes;
    ueNodes.Create (nUEs);
    
    // Setup waypoint mobility
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get (i);
        Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel> ();
        ueNode->AggregateObject (mobility);
        
        // Simple rectangular movement pattern
        double baseX = i * 50.0;
        double baseY = i * 30.0;
        
        // Waypoints in a rectangular pattern
        mobility->AddWaypoint (Waypoint (Seconds (0.0), Vector (baseX, baseY, 0.0)));
        mobility->AddWaypoint (Waypoint (Seconds (5.0), Vector (baseX + 100.0, baseY, 0.0)));
        mobility->AddWaypoint (Waypoint (Seconds (10.0), Vector (baseX + 100.0, baseY + 50.0, 0.0)));
        mobility->AddWaypoint (Waypoint (Seconds (15.0), Vector (baseX, baseY + 50.0, 0.0)));
        mobility->AddWaypoint (Waypoint (Seconds (20.0), Vector (baseX, baseY, 0.0)));
        
        NS_LOG_INFO ("UE " << i << " configured with rectangular movement");
    }
    
    // Install Internet stack
    InternetStackHelper internet;
    internet.Install (ueNodes);
    
    // Simple network
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute ("DataRate", StringValue ("1Mbps"));
    p2p.SetChannelAttribute ("Delay", StringValue ("10ms"));
    
    NetDeviceContainer devices;
    // Create simple line topology
    for (uint32_t i = 0; i < nUEs - 1; ++i)
    {
        NetDeviceContainer link = p2p.Install (ueNodes.Get (i), ueNodes.Get (i + 1));
        devices.Add (link);
    }
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ueInterfaces = ipv4.Assign (devices);
    
    // Enable mobility tracing to file
    std::ofstream traceFile;
    traceFile.open ("shanghai-mobility-trace.txt");
    traceFile << "#Time\tNodeID\tX\tY\tZ" << std::endl;
    traceFile.close ();
    
    // Schedule periodic position logging
    for (uint32_t i = 0; i < nUEs; ++i)
    {
        Simulator::Schedule (Seconds (1.0 + i * 0.1), [i, &ueNodes]() {
            Ptr<Node> ueNode = ueNodes.Get (i);
            Ptr<MobilityModel> mobility = ueNode->GetObject<MobilityModel> ();
            Vector pos = mobility->GetPosition ();
            
            std::ofstream traceFile;
            traceFile.open ("shanghai-mobility-trace.txt", std::ios::app);
            traceFile << Simulator::Now ().GetSeconds () << "\t" << i << "\t" 
                       << pos.x << "\t" << pos.y << "\t" << pos.z << std::endl;
            traceFile.close ();
        });
    }
    
    // Setup logging
    LogComponentEnable ("Shanghai63UEs", LOG_LEVEL_INFO);
    
    NS_LOG_INFO ("Running simulation...");
    
    // Run simulation
    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();
    Simulator::Destroy ();
    
    NS_LOG_INFO ("Shanghai UE Mobility Demo completed successfully");
    NS_LOG_INFO ("Mobility trace saved to: shanghai-mobility-trace.txt");
    
    return 0;
}