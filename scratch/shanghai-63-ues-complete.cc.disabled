/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Complete implementation for 63 UEs with real Shanghai mobility
 * This file demonstrates the full workflow from dataset to NS-3 simulation
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/mmwave-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/flow-monitor-module.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Shanghai63UEs");

// Configuration structure for the 63 UE simulation
struct Shanghai63Config
{
  std::string datasetPath = "/path/to/shanghai/dataset";
  std::string outputPath = "./results/shanghai-63-ues";
  int targetUECount = 63;
  double simulationTime = 60.0;  // seconds
  double timeScale = 1.0;        // Real-time playback
  bool enableTracing = true;
  bool enableFlowMonitor = true;
  std::string mobilityApproach = "waypoint"; // "waypoint", "ns2", or "hybrid"
};

// Main function demonstrating the complete workflow
int main (int argc, char *argv[])
{
  Shanghai63Config config;
  
  // Parse command line arguments
  CommandLine cmd (__FILE__);
  cmd.AddValue ("datasetPath", "Path to Shanghai dataset", config.datasetPath);
  cmd.AddValue ("outputPath", "Output directory for results", config.outputPath);
  cmd.AddValue ("simulationTime", "Simulation time in seconds", config.simulationTime);
  cmd.AddValue ("timeScale", "Time scale for trajectory playback", config.timeScale);
  cmd.AddValue ("mobilityApproach", "Mobility approach (waypoint/ns2/hybrid)", config.mobilityApproach);
  cmd.AddValue ("enableTracing", "Enable detailed tracing", config.enableTracing);
  cmd.Parse (argc, argv);
  
  // Enable logging
  LogComponentEnable ("Shanghai63UEs", LOG_LEVEL_INFO);
  
  NS_LOG_INFO ("Starting Shanghai 63 UE simulation");
  NS_LOG_INFO ("Dataset path: " << config.datasetPath);
  NS_LOG_INFO ("Output path: " << config.outputPath);
  NS_LOG_INFO ("Mobility approach: " << config.mobilityApproach);
  
  // Step 1: Discover and parse scenario files
  std::vector<std::string> scenarioFiles;
  std::string scenarioPattern = config.datasetPath + "/*/*.xosc";
  
  // For demonstration, we'll use a hardcoded list of scenarios
  // In practice, you would scan the directory
  scenarioFiles = {
    config.datasetPath + "/scenario_001/scenario_001.xosc",
    config.datasetPath + "/scenario_002/scenario_002.xosc",
    config.datasetPath + "/scenario_003/scenario_003.xosc",
    config.datasetPath + "/scenario_004/scenario_004.xosc",
    config.datasetPath + "/scenario_005/scenario_005.xosc",
    // Add more scenarios as needed
  };
  
  NS_LOG_INFO ("Found " << scenarioFiles.size () << " scenario files");
  
  // Step 2: Parse scenarios and select 63 UEs
  std::vector<SelectedUE> selectedUEs;
  UESelectionCriteria criteria;
  criteria.minTrajectoryDuration = 10.0;  // 10 seconds minimum
  criteria.minAverageSpeed = 1.0;          // 1 m/s minimum
  criteria.maxAverageSpeed = 30.0;         // 30 m/s maximum
  criteria.minDistance = 50.0;             // 50 meters minimum
  criteria.maxVehiclesPerScenario = 15;   // Distribute across scenarios
  criteria.prioritizeDiverseScenarios = true;
  criteria.prioritizeActiveVehicles = true;
  
  // Create UE selector and select vehicles
  Ptr<ShanghaiUeSelector> ueSelector = CreateObject<ShanghaiUeSelector> ();
  ueSelector->SetSelectionCriteria (criteria);
  selectedUEs = ueSelector->SelectUEs (scenarioFiles, config.targetUECount);
  
  NS_LOG_INFO ("Selected " << selectedUEs.size () << " UEs from " << scenarioFiles.size () << " scenarios");
  
  // Print selection summary
  ueSelector->PrintSelectionSummary (selectedUEs);
  
  // Step 3: Create NS-3 simulation topology
  // Create base stations (e.g., 7 cells in hexagonal layout)
  NodeContainer bsNodes;
  bsNodes.Create (7);
  
  // Create UEs
  NodeContainer ueNodes;
  ueNodes.Create (config.targetUECount);
  
  // Step 4: Configure LTE/mmWave network
  // Use mmWave for high-frequency scenarios
  Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper> ();
  mmwaveHelper->SetSchedulerType ("ns3::MmWaveFlexTtiMacScheduler");
  mmwaveHelper->SetPathlossModelType ("ns3::MmWave3gppBuildingsPropagationLossModel");
  
  // Install mmWave stack
  NetDeviceContainer bsDevices = mmwaveHelper->InstallEnbDevice (bsNodes);
  NetDeviceContainer ueDevices = mmwaveHelper->InstallUeDevice (ueNodes);
  
  // Configure base stations
  MobilityHelper bsMobility;
  bsMobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  
  // Place base stations in hexagonal layout
  Ptr<ListPositionAllocator> bsPositionAlloc = CreateObject<ListPositionAllocator> ();
  double cellRadius = 500.0;  // 500 meters
  double centerX = 1000.0, centerY = 1000.0;
  
  // Center cell
  bsPositionAlloc->Add (Vector (centerX, centerY, 30.0));
  
  // Surrounding 6 cells in hexagonal pattern
  for (int i = 0; i < 6; ++i)
    {
      double angle = i * M_PI / 3.0;
      double x = centerX + cellRadius * std::cos (angle);
      double y = centerY + cellRadius * std::sin (angle);
      bsPositionAlloc->Add (Vector (x, y, 30.0));
    }
  
  bsMobility.SetPositionAllocator (bsPositionAlloc);
  bsMobility.Install (bsNodes);
  
  // Step 5: Configure UE mobility based on selected approach
  MobilityConfiguration mobilityConfig;
  mobilityConfig.implementation = (config.mobilityApproach == "waypoint") ? WAYPOINT_MODEL :
                                 (config.mobilityApproach == "ns2") ? NS2_MOBILITY_HELPER : HYBRID_APPROACH;
  mobilityConfig.timeScale = config.timeScale;
  mobilityConfig.initialDelay = 1.0;
  mobilityConfig.enableCourseChangeTracing = config.enableTracing;
  mobilityConfig.traceFilePath = config.outputPath + "/mobility-trace.txt";
  mobilityConfig.waypointUpdateInterval = 0.1;
  mobilityConfig.synchronizeStartTime = true;
  
  // Create mobility manager
  Ptr<ShanghaiMobilityManager> mobilityManager = CreateObject<ShanghaiMobilityManager> ();
  mobilityManager->SetMobilityConfiguration (mobilityConfig);
  mobilityManager->LoadUEAssignments (selectedUEs);
  
  // Setup mobility based on approach
  if (config.mobilityApproach == "waypoint")
    {
      NS_LOG_INFO ("Using WaypointMobilityModel approach");
      mobilityManager->SetupWaypointMobility (ueNodes);
    }
  else if (config.mobilityApproach == "ns2")
    {
      NS_LOG_INFO ("Using Ns2MobilityHelper approach");
      std::string ns2TraceFile = config.outputPath + "/shanghai-63-ues.ns2";
      mobilityManager->SetupNs2Mobility (ueNodes, ns2TraceFile);
    }
  else
    {
      NS_LOG_INFO ("Using hybrid approach");
      std::string ns2TraceFile = config.outputPath + "/shanghai-63-ues.ns2";
      mobilityManager->SetupHybridMobility (ueNodes, ns2TraceFile);
    }
  
  // Step 6: Attach UEs to base stations
  mmwaveHelper->AttachToClosestEnb (ueDevices, bsDevices);
  mmwaveHelper->ActivateDataRadioBearer (ueDevices);
  
  // Step 7: Install Internet stack
  InternetStackHelper internet;
  internet.Install (ueNodes);
  internet.Install (bsNodes);
  
  // Configure IP addresses
  Ipv4Helper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer ueInterfaces = ipv4.Assign (ueDevices);
  Ipv4InterfaceContainer bsInterfaces = ipv4.Assign (bsDevices);
  
  // Step 8: Install applications for traffic generation
  uint16_t port = 9;  // Discard port
  ApplicationContainer serverApps;
  ApplicationContainer clientApps;
  
  // Install UDP echo servers on base stations
  for (uint32_t i = 0; i < bsNodes.GetN (); ++i)
    {
      UdpEchoServerHelper echoServer (port + i);
      echoServer.SetAttribute ("MaxPackets", UintegerValue (1000));
      echoServer.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
      echoServer.SetAttribute ("PacketSize", UintegerValue (1024));
      
      serverApps.Add (echoServer.Install (bsNodes.Get (i)));
    }
  
  // Install UDP echo clients on UEs
  for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
      UdpEchoClientHelper echoClient (bsInterfaces.GetAddress (i % bsNodes.GetN ()), port + (i % bsNodes.GetN ()));
      echoClient.SetAttribute ("MaxPackets", UintegerValue (100));
      echoClient.SetAttribute ("Interval", TimeValue (Seconds (2.0)));
      echoClient.SetAttribute ("PacketSize", UintegerValue (512));
      echoClient.SetAttribute ("StartTime", TimeValue (Seconds (1.0 + i * 0.1)));
      
      clientApps.Add (echoClient.Install (ueNodes.Get (i)));
    }
  
  serverApps.Start (Seconds (0.0));
  serverApps.Stop (Seconds (config.simulationTime));
  clientApps.Start (Seconds (1.0));
  clientApps.Stop (Seconds (config.simulationTime));
  
  // Step 9: Configure monitoring and tracing
  Ptr<FlowMonitor> flowMonitor;
  FlowMonitorHelper flowHelper;
  
  if (config.enableFlowMonitor)
    {
      flowMonitor = flowHelper.InstallAll ();
    }
  
  // Enable course change tracing for mobility
  if (config.enableTracing)
    {
      AsciiTraceHelper ascii;
      std::string mobilityTraceFile = config.outputPath + "/mobility-course-changes.txt";
      Ptr<OutputStreamWrapper> mobilityStream = ascii.CreateFileStream (mobilityTraceFile);
      
      for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
        {
          ueNodes.Get (i)->GetObject<MobilityModel> ()->TraceConnectWithoutContext (
            "CourseChange", MakeBoundCallback (&CourseChangeTrace, mobilityStream, i));
        }
    }
  
  // Step 10: Run simulation
  NS_LOG_INFO ("Starting simulation for " << config.simulationTime << " seconds");
  
  // Start mobility
  mobilityManager->StartMobility ();
  
  Simulator::Stop (Seconds (config.simulationTime));
  Simulator::Run ();
  
  // Step 11: Collect and analyze results
  if (config.enableFlowMonitor)
    {
      flowMonitor->CheckForLostPackets ();
      flowMonitor->SerializeToXmlFile (config.outputPath + "/flow-monitor.xml", true, true);
      
      // Print flow statistics
      std::map<FlowId, FlowMonitor::FlowStats> stats = flowMonitor->GetFlowStats ();
      NS_LOG_INFO ("Flow Statistics:");
      for (auto& flow : stats)
        {
          NS_LOG_INFO ("Flow " << flow.first << ": "
                      << "Tx packets: " << flow.second.txPackets
                      << ", Rx packets: " << flow.second.rxPackets
                      << ", Lost packets: " << flow.second.lostPackets
                      << ", Throughput: " << flow.second.rxBytes * 8.0 / flow.second.timeLastRxPacket.GetSeconds () / 1024.0 << " Kbps");
        }
    }
  
  // Export mobility summary
  mobilityManager->ExportMobilitySummary (config.outputPath + "/mobility-summary.txt");
  
  // Log final statistics
  std::vector<Vector> finalPositions = mobilityManager->GetCurrentPositions (ueNodes);
  std::vector<Vector> finalVelocities = mobilityManager->GetCurrentVelocities (ueNodes);
  
  NS_LOG_INFO ("Final Statistics:");
  NS_LOG_INFO ("Total UEs: " << ueNodes.GetN ());
  NS_LOG_INFO ("Total base stations: " << bsNodes.GetN ());
  NS_LOG_INFO ("Simulation time: " << config.simulationTime << " seconds");
  NS_LOG_INFO ("Time scale: " << config.timeScale);
  
  // Calculate average distance traveled
  double totalDistance = 0.0;
  for (size_t i = 0; i < selectedUEs.size (); ++i)
    {
      totalDistance += selectedUEs[i].vehicleInfo.totalDistance;
    }
  NS_LOG_INFO ("Average distance per UE: " << totalDistance / selectedUEs.size () << " meters");
  
  Simulator::Destroy ();
  NS_LOG_INFO ("Simulation completed successfully");
  
  return 0;
}

// Course change tracing callback
void CourseChangeTrace (Ptr<OutputStreamWrapper> stream, uint32_t nodeId, Ptr<const MobilityModel> mobility)
{
  Vector pos = mobility->GetPosition ();
  Vector vel = mobility->GetVelocity ();
  *stream->GetStream () << Simulator::Now ().GetSeconds () << " " << nodeId 
                        << " " << pos.x << " " << pos.y << " " << pos.z
                        << " " << vel.x << " " << vel.y << " " << vel.z << std::endl;
}