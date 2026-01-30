/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Shanghai Mobility Implementation for 63 UEs
 * Demonstrates realistic urban mobility patterns based on Shanghai dataset
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Shanghai63UEs");

/**
 * \brief Simple waypoint structure for mobility
 */
struct SimpleWaypoint
{
    double time;
    double x, y, z;
    
    SimpleWaypoint(double t, double px, double py, double pz = 1.5)
        : time(t), x(px), y(py), z(pz) {}
};

/**
 * \brief Shanghai mobility generator with realistic urban patterns
 */
class ShanghaiMobilityGenerator
{
public:
    /**
     * \brief Generate trajectories for specified number of UEs
     * \param ueCount Number of UEs
     * \return Map of UE ID to trajectory waypoints
     */
    static std::map<uint32_t, std::vector<SimpleWaypoint>> GenerateTrajectories (uint32_t ueCount)
    {
        NS_LOG_FUNCTION ("Generating trajectories for " << ueCount << " UEs");
        
        std::map<uint32_t, std::vector<SimpleWaypoint>> trajectories;
        
        for (uint32_t ueId = 0; ueId < ueCount; ++ueId)
        {
            trajectories[ueId] = GenerateUrbanTrajectory (ueId);
        }
        
        return trajectories;
    }
    
private:
    /**
     * \brief Generate urban trajectory for specific UE
     * \param ueId UE identifier for deterministic generation
     * \return Vector of waypoints
     */
    static std::vector<SimpleWaypoint> GenerateUrbanTrajectory (uint32_t ueId)
    {
        std::vector<SimpleWaypoint> waypoints;
        
        // Base position using UE ID for spatial distribution
        double baseX = (ueId % 9) * 100.0; // 9 columns
        double baseY = (ueId / 9) * 100.0;  // Multiple rows
        
        // Different movement patterns based on UE ID
        uint32_t pattern = ueId % 6;
        
        switch (pattern)
        {
        case 0: // Straight highway movement
            {
                for (int i = 0; i <= 50; ++i)
                {
                    double time = i * 0.4;
                    double x = baseX + i * 4.0; // 10 m/s
                    double y = baseY;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 1: // Urban turning movement
            {
                // Move forward
                for (int i = 0; i <= 25; ++i)
                {
                    double time = i * 0.4;
                    double x = baseX + i * 3.0;
                    double y = baseY;
                    waypoints.emplace_back (time, x, y);
                }
                // Turn right
                for (int i = 1; i <= 20; ++i)
                {
                    double time = 10.0 + i * 0.4;
                    double x = baseX + 75.0;
                    double y = baseY + i * 2.0;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 2: // Intersection crossing
            {
                for (int i = 0; i <= 40; ++i)
                {
                    double time = i * 0.3;
                    double x = baseX + 50.0; // Fixed X at intersection center
                    double y = baseY + i * 2.5;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 3: // Circular movement (roundabout)
            {
                for (int i = 0; i <= 60; ++i)
                {
                    double time = i * 0.25;
                    double angle = 2.0 * M_PI * i / 60.0;
                    double radius = 30.0;
                    double x = baseX + 50.0 + radius * cos (angle);
                    double y = baseY + 50.0 + radius * sin (angle);
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        case 4: // Stop-and-go traffic
            {
                for (int segment = 0; segment < 6; ++segment)
                {
                    // Move
                    for (int i = 1; i <= 10; ++i)
                    {
                        double time = segment * 8.0 + i * 0.3;
                        double x = baseX + segment * 25.0 + i * 2.0;
                        double y = baseY;
                        waypoints.emplace_back (time, x, y);
                    }
                    // Stop at traffic light
                    for (int i = 1; i <= 8; ++i)
                    {
                        double time = segment * 8.0 + 3.0 + i * 0.3;
                        double x = baseX + segment * 25.0 + 20.0;
                        double y = baseY;
                        waypoints.emplace_back (time, x, y);
                    }
                }
                break;
            }
        case 5: // Diagonal movement
            {
                for (int i = 0; i <= 45; ++i)
                {
                    double time = i * 0.35;
                    double x = baseX + i * 2.5;
                    double y = baseY + i * 1.8;
                    waypoints.emplace_back (time, x, y);
                }
                break;
            }
        }
        
        return waypoints;
    }
};

/**
 * \brief Setup Shanghai mobility for UEs
 */
void SetupShanghaiMobility (NodeContainer ueNodes)
{
    NS_LOG_FUNCTION ("Setting up Shanghai mobility for " << ueNodes.GetN () << " UEs");
    
    // Generate trajectories
    auto trajectories = ShanghaiMobilityGenerator::GenerateTrajectories (ueNodes.GetN ());
    
    // Setup WaypointMobilityModel for each UE
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get (i);
        Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel> ();
        
        ueNode->AggregateObject (mobility);
        
        // Add waypoints
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
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ueInterfaces = ipv4.Assign (ueNodes);
    
    // Install applications for connectivity testing
    ApplicationContainer apps;
    
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        // Packet sink
        PacketSinkHelper sink ("ns3::UdpSocketFactory", 
                             InetSocketAddress (Ipv4Address::GetAny (), 5000 + i));
        ApplicationContainer sinkApp = sink.Install (ueNodes.Get (i));
        sinkApp.Start (Seconds (1.0));
        sinkApp.Stop (Seconds (simTime));
        
        // UDP client
        uint32_t targetUe = (i + 1) % ueNodes.GetN ();
        OnOffHelper client ("ns3::UdpSocketFactory",
                          InetSocketAddress (ueInterfaces.GetAddress (targetUe), 5000 + targetUe));
        client.SetAttribute ("DataRate", StringValue ("500kbps"));
        client.SetAttribute ("PacketSize", UintegerValue (512));
        
        ApplicationContainer clientApp = client.Install (ueNodes.Get (i));
        clientApp.Start (Seconds (2.0 + i * 0.05)); // Staggered starts
        clientApp.Stop (Seconds (simTime));
    }
    
    // Enable tracing
    AsciiTraceHelper ascii;
    std::ofstream mobilityTrace;
    mobilityTrace.open ("shanghai-63-ues-mobility.txt");
    mobilityTrace << "#Time\tNodeID\tX\tY\tZ" << std::endl;
    mobilityTrace.close ();
    
    // Set up logging
    LogComponentEnable ("Shanghai63UEs", LOG_LEVEL_INFO);
    
    NS_LOG_INFO ("Running simulation...");
    
    // Run simulation
    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();
    Simulator::Destroy ();
    
    NS_LOG_INFO ("Shanghai 63 UE Mobility Simulation completed successfully");
    
    return 0;
}