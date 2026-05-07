/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Shanghai 63 UE Mobility Demo - Fixed Version
 * Demonstrates urban mobility patterns for 63 UEs
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include <fstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Shanghai63UEs");

/**
 * \brief Generate Shanghai-like urban mobility for 63 UEs
 */
void SetupShanghaiMobility (NodeContainer ueNodes)
{
    NS_LOG_FUNCTION ("Setting up Shanghai mobility for " << ueNodes.GetN () << " UEs");
    
    // Different movement patterns for UEs
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get (i);
        Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel> ();
        ueNode->AggregateObject (mobility);
        
        // Base position for spatial distribution
        double baseX = (i % 9) * 100.0; // 9 columns
        double baseY = (i / 9) * 100.0;  // Multiple rows
        
        uint32_t pattern = i % 6;
        
        switch (pattern)
        {
        case 0: // Straight highway
            {
                for (int j = 0; j <= 30; ++j)
                {
                    double time = j * 0.4;
                    double x = baseX + j * 5.0;
                    double y = baseY;
                    Vector pos (x, y, 1.5);
                    mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                }
                break;
            }
        case 1: // Urban turning
            {
                // Forward
                for (int j = 0; j <= 20; ++j)
                {
                    double time = j * 0.4;
                    double x = baseX + j * 3.0;
                    double y = baseY;
                    Vector pos (x, y, 1.5);
                    mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                }
                // Turn right
                for (int j = 1; j <= 15; ++j)
                {
                    double time = 8.0 + j * 0.4;
                    double x = baseX + 60.0;
                    double y = baseY + j * 2.5;
                    Vector pos (x, y, 1.5);
                    mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                }
                break;
            }
        case 2: // Intersection crossing
            {
                for (int j = 0; j <= 30; ++j)
                {
                    double time = j * 0.3;
                    double x = baseX + 50.0;
                    double y = baseY + j * 3.0;
                    Vector pos (x, y, 1.5);
                    mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                }
                break;
            }
        case 3: // Roundabout
            {
                for (int j = 0; j <= 40; ++j)
                {
                    double time = j * 0.25;
                    double angle = 2.0 * 3.1415926535 * j / 40.0;
                    double radius = 25.0;
                    double x = baseX + 50.0 + radius * std::cos (angle);
                    double y = baseY + 50.0 + radius * std::sin (angle);
                    Vector pos (x, y, 1.5);
                    mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                }
                break;
            }
        case 4: // Stop-and-go
            {
                for (int segment = 0; segment < 5; ++segment)
                {
                    // Move
                    for (int j = 1; j <= 8; ++j)
                    {
                        double time = segment * 8.0 + j * 0.3;
                        double x = baseX + segment * 30.0 + j * 2.5;
                        double y = baseY;
                        Vector pos (x, y, 1.5);
                        mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                    }
                    // Stop
                    for (int j = 1; j <= 6; ++j)
                    {
                        double time = segment * 8.0 + 2.4 + j * 0.3;
                        double x = baseX + segment * 30.0 + 20.0;
                        double y = baseY;
                        Vector pos (x, y, 1.5);
                        mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                    }
                }
                break;
            }
        case 5: // Diagonal
            {
                for (int j = 0; j <= 25; ++j)
                {
                    double time = j * 0.35;
                    double x = baseX + j * 4.0;
                    double y = baseY + j * 2.5;
                    Vector pos (x, y, 1.5);
                    mobility->AddWaypoint (Waypoint (Seconds (time), pos));
                }
                break;
            }
        }
        
        NS_LOG_INFO ("UE " << i << " configured with pattern " << pattern);
    }
    
    NS_LOG_INFO ("Shanghai mobility setup completed");
}

int 
main (int argc, char *argv[])
{
    // Parameters
    double simTime = 60.0;
    uint32_t nUEs = 63;
    
    CommandLine cmd;
    cmd.AddValue ("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue ("nUEs", "Number of UEs", nUEs);
    cmd.Parse (argc, argv);
    
    NS_LOG_INFO ("Starting Shanghai 63 UE Mobility Simulation");
    NS_LOG_INFO ("Simulation time: " << simTime << " seconds");
    NS_LOG_INFO ("Number of UEs: " << nUEs);
    
    // Create UE nodes
    NodeContainer ueNodes;
    ueNodes.Create (nUEs);
    
    // Setup Shanghai mobility
    SetupShanghaiMobility (ueNodes);
    
    // Install Internet stack
    InternetStackHelper internet;
    internet.Install (ueNodes);
    
    // Create simple point-to-point network
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute ("DataRate", StringValue ("1Mbps"));
    p2p.SetChannelAttribute ("Delay", StringValue ("5ms"));
    
    NetDeviceContainer devices;
    // Connect UEs in a simple chain topology
    for (uint32_t i = 0; i < nUEs - 1; ++i)
    {
        NetDeviceContainer link = p2p.Install (ueNodes.Get (i), ueNodes.Get (i + 1));
        devices.Add (link);
    }
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ueInterfaces = ipv4.Assign (devices);
    
    // Install applications
    for (uint32_t i = 0; i < nUEs; ++i)
    {
        // Packet sink
        PacketSinkHelper sink ("ns3::UdpSocketFactory", 
                             InetSocketAddress (Ipv4Address::GetAny (), 5000 + i));
        ApplicationContainer sinkApp = sink.Install (ueNodes.Get (i));
        sinkApp.Start (Seconds (1.0));
        sinkApp.Stop (Seconds (simTime));
        
        // Simple UDP client
        if (i < nUEs - 1)  // Last UE doesn't send
        {
            OnOffHelper client ("ns3::UdpSocketFactory",
                              InetSocketAddress (ueInterfaces.GetAddress (i + 1), 5000 + i + 1));
            client.SetAttribute ("DataRate", StringValue ("50kbps"));
            client.SetAttribute ("PacketSize", UintegerValue (128));
            
            ApplicationContainer clientApp = client.Install (ueNodes.Get (i));
            clientApp.Start (Seconds (2.0 + i * 0.1));  // Staggered start
            clientApp.Stop (Seconds (simTime));
        }
    }
    
    // Enable tracing
    MobilityHelper::EnableAsciiAll ("shanghai-63-ues-mobility.tr");
    
    // Set up logging
    LogComponentEnable ("Shanghai63UEs", LOG_LEVEL_INFO);
    
    NS_LOG_INFO ("Running simulation...");
    
    // Run simulation
    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();
    Simulator::Destroy ();
    
    NS_LOG_INFO ("Shanghai 63 UE Mobility Simulation completed successfully");
    NS_LOG_INFO ("Mobility trace file: shanghai-63-ues-mobility.tr");
    
    return 0;
}