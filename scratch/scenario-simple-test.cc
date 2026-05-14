/*
 * scenario-simple-test.cc
 * A lightweight test scenario to verify 100ms metrics and RLF trackers
 * for O-RAN offline DRL agents.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include <ns3/lte-ue-net-device.h>
#include "ns3/mmwave-helper.h"
#include "ns3/epc-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/lte-helper.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <cmath>
#include <string>

using namespace ns3;
using namespace mmwave;

NS_LOG_COMPONENT_DEFINE ("ScenarioSimpleTest");

// --- 1. RLF Tracker Global Variables ---
static std::map<uint16_t, std::set<uint64_t>> g_badUesPerCell;

static void
RlfSinrCb(std::string path, uint16_t cellId, uint16_t rnti, double rsrp, double sinrLinear, uint8_t componentCarrierId)
{
    (void) path;
    (void) rsrp;
    (void) componentCarrierId;
    double sinrDb = 10.0 * std::log10(sinrLinear);
    if (sinrDb < -5.0) {
        g_badUesPerCell[cellId].insert(rnti);
    }
}

static void
MmWaveRlfSinrCb(std::string path, uint64_t imsi, uint16_t cellId, long double sinrLinear)
{
    (void) path;
    double sinrDb = 10.0 * std::log10(static_cast<double>(sinrLinear));
    if (sinrDb < -5.0) {
        g_badUesPerCell[cellId].insert(imsi);
    }
}

static void
DumpRlfCounters()
{
    for (const auto& kv : g_badUesPerCell) {
        std::cout << "RLF_DUMP," << Simulator::Now().GetSeconds() 
                  << "," << kv.first << "," << kv.second.size() << std::endl;
    }
    g_badUesPerCell.clear();
    Simulator::Schedule(MilliSeconds(100), &DumpRlfCounters);
}

// --- 2. BS State Tracker ---
std::ofstream outFile;
void
BsStateTrace (std::string filename, Ptr<LteEnbNetDevice> ltedev, Ptr<LteEnbRrc> lte_rrc)
{
    if (!outFile.is_open ()) {
        outFile.open (filename.c_str (), std::ios_base::out | std::ios_base::trunc);
        outFile << "Timestamp UNIX Id State\n";
    }
    std::map<uint16_t, bool> entry = lte_rrc->GetAllowHandoverTo();
    for (auto it = entry.begin(); it != entry.end(); it++) {
        uint64_t unix_timestamp_ms = ltedev->GetStartTime() + Simulator::Now ().GetMilliSeconds ();
        outFile << Simulator::Now ().GetSeconds () << " " << unix_timestamp_ms << " "
                << it->first << " " << it->second << std::endl;
    }
}

int
main (int argc, char *argv[])
{
    double simTime = 2.0; 
    double indicationPeriodicity = 0.1; // 100ms
    bool harqEnabled = true;
    bool rlcAmEnabled = true;
    bool useSemaphores = false;
    bool scheduleControlMessages = false;
    bool enableSimpleMobility = false;
    std::string controlFilename = "";
    uint16_t nUeNodes = 2;
    uint16_t nMmWaveEnbNodes = 2;
    uint16_t nLteEnbNodes = 1;

    CommandLine cmd;
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("useSemaphores", "If true, enables semaphore-based external control", useSemaphores);
    cmd.AddValue("controlFileName", "Control payload file for hierarchical-style actions", controlFilename);
    cmd.AddValue("scheduleControlMessages",
                 "If true, execute control actions at the timestamp written in the file",
                 scheduleControlMessages);
    cmd.AddValue("enableSimpleMobility",
                 "If true, keep the simple deterministic UE mobility enabled",
                 enableSimpleMobility);
    cmd.Parse (argc, argv);

    //std::cout << "### controlFilename: " << controlFilename << "### " << std::endl;

    if (controlFilename == "none"){
        std::cout << "No control file provided, running with default settings." << std::endl;
        controlFilename = "";
    }
    

    // O-RAN E2 Logging defaults
    Config::SetDefault ("ns3::LteEnbNetDevice::UseSemaphores", BooleanValue (useSemaphores));
    Config::SetDefault ("ns3::LteEnbNetDevice::ControlFileName", StringValue (controlFilename));
    Config::SetDefault ("ns3::LteEnbNetDevice::ScheduleControlMessages",
                        BooleanValue (scheduleControlMessages));
    Config::SetDefault ("ns3::LteEnbNetDevice::E2Periodicity", DoubleValue (indicationPeriodicity));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::E2Periodicity", DoubleValue (indicationPeriodicity));
    Config::SetDefault ("ns3::MmWaveHelper::E2Periodicity", DoubleValue (indicationPeriodicity));
    Config::SetDefault ("ns3::MmWaveHelper::E2ModeLte", BooleanValue(true));
    Config::SetDefault ("ns3::MmWaveHelper::E2ModeNr", BooleanValue(true));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableDuReport", BooleanValue(true));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableCuUpReport", BooleanValue(true));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableCuCpReport", BooleanValue(true));
    Config::SetDefault ("ns3::LteEnbNetDevice::EnableE2FileLogging", BooleanValue (true));
    Config::SetDefault ("ns3::MmWaveEnbNetDevice::EnableE2FileLogging", BooleanValue (true));
    Config::SetDefault ("ns3::MmWaveHelper::RlcAmEnabled", BooleanValue (rlcAmEnabled));
    Config::SetDefault ("ns3::MmWaveHelper::HarqEnabled", BooleanValue (harqEnabled));
    Config::SetDefault ("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue (harqEnabled));
    Config::SetDefault ("ns3::MmWavePhyMacCommon::NumHarqProcess", UintegerValue (100));
    Config::SetDefault ("ns3::MmWaveHelper::UseIdealRrc", BooleanValue (true));
    Config::SetDefault ("ns3::MmWaveUeMac::UpdateUeSinrEstimatePeriod", DoubleValue (0));
    Config::SetDefault ("ns3::ThreeGppChannelModel::UpdatePeriod",
                        TimeValue (MilliSeconds (100.0)));
    Config::SetDefault ("ns3::ThreeGppChannelConditionModel::UpdatePeriod",
                        TimeValue (MilliSeconds (100.0)));
    Config::SetDefault ("ns3::LteEnbRrc::OutageThreshold", DoubleValue (-5.0));
    Config::SetDefault ("ns3::LteEnbRrc::SystemInformationPeriodicity",
                        TimeValue (MilliSeconds (5.0)));
    Config::SetDefault ("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue (320));
    Config::SetDefault ("ns3::LteEnbRrc::FirstSibTime", UintegerValue (2));
    Config::SetDefault ("ns3::RadioBearerStatsCalculator::EpochDuration",
                        TimeValue (Seconds (indicationPeriodicity)));
    Config::SetDefault ("ns3::MmWaveBearerStatsCalculator::EpochDuration",
                        TimeValue (Seconds (indicationPeriodicity)));
    
    // Ensure fast reporting
    Config::SetDefault ("ns3::LteRlcAm::ReportBufferStatusTimer", TimeValue (MilliSeconds (10.0)));
    Config::SetDefault ("ns3::LteRlcUmLowLat::ReportBufferStatusTimer",
                        TimeValue (MilliSeconds (10.0)));
    Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (1024 * 1024));
    Config::SetDefault ("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue (1024 * 1024));
    Config::SetDefault ("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue (1024 * 1024));

    Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper> ();
    Ptr<MmWavePointToPointEpcHelper> epcHelper = CreateObject<MmWavePointToPointEpcHelper> ();
    mmwaveHelper->SetEpcHelper (epcHelper);

    // Create Nodes
    NodeContainer ueNodes, mmWaveEnbNodes, lteEnbNodes, allEnbNodes;
    ueNodes.Create (nUeNodes);
    mmWaveEnbNodes.Create (nMmWaveEnbNodes);
    lteEnbNodes.Create (nLteEnbNodes);
    allEnbNodes.Add (lteEnbNodes);
    allEnbNodes.Add (mmWaveEnbNodes);

    // EPC / Internet setup
    Ptr<Node> pgw = epcHelper->GetPgwNode ();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create (1);
    Ptr<Node> remoteHost = remoteHostContainer.Get (0);
    InternetStackHelper internet;
    internet.Install (remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("10Gb/s")));
    p2ph.SetDeviceAttribute ("Mtu", UintegerValue (2500));
    p2ph.SetChannelAttribute ("Delay", TimeValue (Seconds (0.010)));
    NetDeviceContainer internetDevices = p2ph.Install (pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase ("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign (internetDevices);

    
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting (remoteHost->GetObject<Ipv4> ());
    remoteHostStaticRouting->AddNetworkRouteTo (Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"), 1);

    // Simple Mobility (Small Area)
    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator> ();
    enbPositionAlloc->Add (Vector (50, 50, 3));  // LTE
    enbPositionAlloc->Add (Vector (50, 50, 3));  // gNB 1
    enbPositionAlloc->Add (Vector (150, 50, 3)); // gNB 2

    MobilityHelper enbmobility;
    enbmobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    enbmobility.SetPositionAllocator (enbPositionAlloc);
    enbmobility.Install (allEnbNodes);

    MobilityHelper uemobility;
    uemobility.SetMobilityModel ("ns3::ConstantVelocityMobilityModel");
    uemobility.Install (ueNodes);

    Ptr<ConstantVelocityMobilityModel> ue0Mobility =
        ueNodes.Get (0)->GetObject<ConstantVelocityMobilityModel> ();
    Ptr<ConstantVelocityMobilityModel> ue1Mobility =
        ueNodes.Get (1)->GetObject<ConstantVelocityMobilityModel> ();
    ue0Mobility->SetPosition (Vector (60, 50, 1.5));
    ue1Mobility->SetPosition (Vector (140, 50, 1.5));

    // Keep UEs stationary by default so control-path tests are isolated from mobility effects.
    if (enableSimpleMobility)
    {
        ue0Mobility->SetVelocity (Vector (260.0, 0.0, 0.0));
        ue1Mobility->SetVelocity (Vector (220.0, 35.0, 0.0));
    }
    else
    {
        ue0Mobility->SetVelocity (Vector (0.0, 0.0, 0.0));
        ue1Mobility->SetVelocity (Vector (0.0, 0.0, 0.0));
    }

    // Install Radio Devices
    NetDeviceContainer lteEnbDevs = mmwaveHelper->InstallLteEnbDevice (lteEnbNodes);
    NetDeviceContainer mmWaveEnbDevs = mmwaveHelper->InstallEnbDevice (mmWaveEnbNodes);
    NetDeviceContainer mcUeDevs = mmwaveHelper->InstallMcUeDevice (ueNodes);

    internet.Install (ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address (NetDeviceContainer (mcUeDevs));

    for (uint32_t u = 0; u < ueNodes.GetN (); ++u) {
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting (ueNodes.Get(u)->GetObject<Ipv4> ());
        ueStaticRouting->SetDefaultRoute (epcHelper->GetUeDefaultGatewayAddress (), 1);
    }
    mmwaveHelper->AddX2Interface (lteEnbNodes, mmWaveEnbNodes);
    mmwaveHelper->AttachToClosestEnb (mcUeDevs, mmWaveEnbDevs, lteEnbDevs);

    // Simple UDP Traffic (Full Buffer)
    uint16_t portUdp = 60000;
    ApplicationContainer sinkApp, clientApp;
    for (uint32_t u = 0; u < ueNodes.GetN (); ++u) {
        PacketSinkHelper dlPacketSinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), portUdp));
        sinkApp.Add (dlPacketSinkHelper.Install (ueNodes.Get (u)));
        
        OnOffHelper dlClient ("ns3::UdpSocketFactory", InetSocketAddress (ueIpIface.GetAddress (u), portUdp));
        dlClient.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=100.0]"));
        dlClient.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0.0]"));
        dlClient.SetAttribute ("DataRate", StringValue ("2Mbps"));
        dlClient.SetAttribute ("PacketSize", UintegerValue (1024));
        clientApp.Add (dlClient.Install (remoteHost));
    }
    sinkApp.Start (Seconds (0.1));
    clientApp.Start (Seconds (0.2));
    clientApp.Stop (Seconds (simTime - 0.1));

    // --- 3. APPLY METRIC FIXES ---
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
    lteHelper->Initialize ();
    lteHelper->EnablePhyTraces ();
    lteHelper->EnableMacTraces ();
    
    // Force 100ms epochs for RLC/PDCP
    lteHelper->EnablePdcpTraces();
    lteHelper->EnableRlcTraces();

    mmwaveHelper->EnableTraces();

    // Connect the RLF SINR Tracker
    Config::ConnectFailSafe("/NodeList/*/DeviceList/*/ComponentCarrierMapUe/*/LteUePhy/ReportCurrentCellRsrpSinr", MakeCallback(&RlfSinrCb));
    Config::ConnectFailSafe("/NodeList/*/DeviceList/*/LteComponentCarrierMapUe/*/LteUePhy/ReportCurrentCellRsrpSinr", MakeCallback(&RlfSinrCb));
    Config::ConnectFailSafe("/NodeList/*/DeviceList/*/LteEnbRrc/NotifyMmWaveSinr", MakeCallback(&MmWaveRlfSinrCb));
    Simulator::Schedule(MilliSeconds(100), &DumpRlfCounters);

    // Schedule bsState.txt
    Ptr<LteEnbNetDevice> ltedev = DynamicCast<LteEnbNetDevice> (lteEnbDevs.Get (0));
    Ptr<LteEnbRrc> lte_rrc = ltedev->GetRrc ();
    int numSteps = static_cast<int>(std::ceil(simTime / indicationPeriodicity));
    for (int step = 0; step <= numSteps; ++step) {
        Simulator::Schedule(Seconds(step * indicationPeriodicity), BsStateTrace, "bsState.txt", ltedev, lte_rrc);
    }

    std::cout << "Starting Simple Test Scenario for " << simTime
              << " seconds. Control File: '" << controlFilename
              << "' Use Semaphores: " << useSemaphores
              << " Schedule Control Messages: " << scheduleControlMessages
              << " Mobility: " << enableSimpleMobility << std::endl;
    Simulator::Stop (Seconds (simTime));
    Simulator::Run ();

    if (outFile.is_open ()) outFile.close ();
    Simulator::Destroy ();
    return 0;
}
