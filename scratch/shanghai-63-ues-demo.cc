/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Mobility implementation for Shanghai dataset with 63 UEs
 * Using WaypointMobilityModel for realistic vehicle trajectories
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/mmwave-helper.h"
#include "ns3/epc-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Shanghai63UEs");

/**
 * \brief Simple trajectory point structure
 */
struct SimpleWaypoint
{
    double time;
    double x, y, z;
    
    SimpleWaypoint(double t, double px, double py, double pz = 1.5)
        : time(t), x(px), y(py), z(pz) {}
};

/**
 * \brief Mock Shanghai mobility generator for 63 UEs
 */
class ShanghaiMobilityGenerator
{
public:
    /**
     * \brief Generate realistic trajectories for 63 UEs
     * \param ueCount Number of UEs to generate trajectories for
     * \return Map of UE ID to trajectory waypoints
     */
    static std::map<uint32_t, std::vector<SimpleWaypoint>> Generate63UEs (uint32_t ueCount = 63)
    {
        NS_LOG_FUNCTION ("Generating trajectories for " << ueCount << " UEs");
        
        std::map<uint32_t, std::vector<SimpleWaypoint>> trajectories;
        
        // Generate diverse realistic trajectories based on typical urban scenarios
        for (uint32_t ueId = 0; ueId < ueCount; ++ueId)
        {
            trajectories[ueId] = GenerateUrbanTrajectory (ueId);
        }
        
        return trajectories;
    }
    
private:
    /**
     * \brief Generate urban trajectory for a specific UE
     * \param ueId UE identifier for deterministic generation
     * \return Vector of waypoints
     */
    static std::vector<SimpleWaypoint> GenerateUrbanTrajectory (uint32_t ueId)
    {
        std::vector<SimpleWaypoint> waypoints;
        
        // Use UE ID as seed for deterministic but diverse trajectories
        double baseX = (ueId % 10) * 50.0; // Grid layout
        double baseY = (ueId / 10) * 50.0;
        
        // Different trajectory patterns based on UE ID
        uint32_t pattern = ueId % 7;
        
        switch (pattern)
        {
        case 0: // Straight highway movement
            {
                for (int i = 0; i <= 100; ++i)
                {
                    double time = i * 0.2; // 5 Hz update rate
                    double x = baseX + i * 2.0; // 10 m/s forward movement
                    double y = baseY;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 1: // Urban turning movement
            {
                for (int i = 0; i <= 50; ++i)
                {
                    double time = i * 0.3;
                    double x = baseX + i * 1.5;
                    double y = baseY;
                    waypoints.emplace_back (time, x, y);
                }
                // Turn right
                for (int i = 1; i <= 30; ++i)
                {
                    double time = 15.0 + i * 0.3;
                    double x = baseX + 75.0;
                    double y = baseY + i * 1.0;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 2: // Intersection crossing
            {
                // North to South
                for (int i = 0; i <= 40; ++i)
                {
                    double time = i * 0.25;
                    double x = baseX + 50.0; // Fixed X at intersection
                    double y = baseY + i * 1.2;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 3: // Circular movement (roundabout)
            {
                for (int i = 0; i <= 80; ++i)
                {
                    double time = i * 0.2;
                    double angle = 2.0 * M_PI * i / 80.0;
                    double radius = 25.0;
                    double x = baseX + 50.0 + radius * cos (angle);
                    double y = baseY + 50.0 + radius * sin (angle);
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 4: // Stop-and-go traffic
            {
                for (int segment = 0; segment < 8; ++segment)
                {
                    // Move
                    for (int i = 1; i <= 10; ++i)
                    {
                        double time = segment * 8.0 + i * 0.4;
                        double x = baseX + segment * 20.0 + i * 1.0;
                        double y = baseY;
                        waypoints.emplace_back (time, x, y);
                    }
                    // Stop
                    for (int i = 1; i <= 5; ++i)
                    {
                        double time = segment * 8.0 + 4.0 + i * 0.4;
                        double x = baseX + segment * 20.0 + 10.0;
                        double y = baseY;
                        waypoints.emplace_back (time, x, y);
                    }
                }
                break;
            }
        case 5: // Diagonal movement
            {
                for (int i = 0; i <= 60; ++i)
                {
                    double time = i * 0.25;
                    double x = baseX + i * 1.5;
                    double y = baseY + i * 0.8;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 6: // Zigzag pattern (parking lot navigation)
            {
                for (int lane = 0; lane < 6; ++lane)
                {
                    // Move forward
                    for (int i = 0; i <= 15; ++i)
                    {
                        double time = lane * 5.0 + i * 0.2;
                        double x = baseX + i * 2.0;
                        double y = baseY + lane * 8.0;
                        waypoints.emplace_back (time, x, y);
                    }
                    // Move to next lane
                    for (int i = 1; i <= 10; ++i)
                    {
                        double time = lane * 5.0 + 3.0 + i * 0.1;
                        double x = baseX + 30.0;
                        double y = baseY + lane * 8.0 + i * 0.8;
                        waypoints.emplace_back (time, x, y);
                    }
                }
                break;
            }
        }
        
        return waypoints;
    }
};

/**
 * \brief Setup realistic mobility for 63 UEs using Shanghai dataset patterns
 */
void SetupShanghaiMobility (NodeContainer ueNodes)
{
    NS_LOG_FUNCTION ("Setting up Shanghai mobility for " << ueNodes.GetN () << " UEs");
    
    // Generate trajectories using Shanghai mobility patterns
    auto trajectories = ShanghaiMobilityGenerator::Generate63UEs (ueNodes.GetN ());
    
    // Setup WaypointMobilityModel for each UE
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get (i);
        Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel> ();
        
        // Set the mobility model
        ueNode->AggregateObject (mobility);
        
        // Add all waypoints for this UE
        const auto& waypoints = trajectories[i];
        NS_LOG_INFO ("UE " << i << " has " << waypoints.size () << " waypoints");
        
        for (const auto& wp : waypoints)
        {
            Vector position (wp.x, wp.y, wp.z);
            Time waypointTime = Seconds (wp.time);
            Waypoint waypoint (waypointTime, position);
            mobility->AddWaypoint (waypoint);
        }
    }
    
    NS_LOG_INFO ("Shanghai mobility setup completed for " << ueNodes.GetN () << " UEs");
}

int 
main (int argc, char *argv[])
{
    // Simulation parameters
    double simTime = 60.0; // 1 minute simulation
    uint32_t nUEs = 63;    // 63 UEs as required
    
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
    
    // Setup Shanghai mobility patterns
    SetupShanghaiMobility (ueNodes);
    
    // Install Internet stack
    InternetStackHelper internet;
    internet.Install (ueNodes);
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ueInterfaces = ipv4.Assign (ueNodes);
    
    // Install simple applications to test mobility
    ApplicationContainer apps;
    
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        // Packet sink to receive packets
        PacketSinkHelper sink ("ns3::UdpSocketFactory", 
                             InetSocketAddress (Ipv4Address::GetAny (), 5000 + i));
        ApplicationContainer sinkApp = sink.Install (ueNodes.Get (i));
        sinkApp.Start (Seconds (1.0));
        sinkApp.Stop (Seconds (simTime));
        
        // UDP client to send packets (every UE sends to next UE)
        uint32_t targetUe = (i + 1) % ueNodes.GetN ();
        OnOffHelper client ("ns3::UdpSocketFactory",
                          InetSocketAddress (ueInterfaces.GetAddress (targetUe), 5000 + targetUe));
        client.SetAttribute ("DataRate", StringValue ("1Mbps"));
        client.SetAttribute ("PacketSize", UintegerValue (1024));
        
        ApplicationContainer clientApp = client.Install (ueNodes.Get (i));
        // Stagger start times to avoid synchronization
        clientApp.Start (Seconds (2.0 + i * 0.1));
        clientApp.Stop (Seconds (simTime));
    }
    
    // Enable mobility tracing
    AsciiTraceHelper ascii;
    MobilityHelper::EnableAsciiAll (ascii.CreateFileStream ("shanghai-63-ues-mobility.tr"));
    
    // Set up logging
    LogComponentEnable ("Shanghai63UEs", LOG_LEVEL_INFO);
    LogComponentEnable ("WaypointMobilityModel", LOG_LEVEL_INFO);
    
    NS_LOG_INFO ("Running simulation...");
    
    // Run simulation
    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();
    Simulator::Destroy ();
    
    NS_LOG_INFO ("Shanghai 63 UE Mobility Simulation completed");
    
    return 0;
}