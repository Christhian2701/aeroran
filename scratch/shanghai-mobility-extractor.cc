#include "shanghai-mobility-extractor.h"
#include "ns3/log.h"
#include "ns3/waypoint-mobility-model.h"
#include "ns3/ns2-mobility-helper.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ShanghaiMobilityExtractor");

std::map<std::string, std::vector<VehicleTrajectory>> 
ShanghaiMobilityExtractor::ParseAllScenarios (const std::string& datasetPath)
{
    NS_LOG_FUNCTION (datasetPath);
    
    std::map<std::string, std::vector<VehicleTrajectory>> allTrajectories;
    std::string scenariosPath = datasetPath + "/Cityvillage/20240121_530_scens/summary_new_15_scens/";
    
    // Check if directory exists
    if (!std::filesystem::exists (scenariosPath))
    {
        NS_LOG_ERROR ("Scenarios directory not found: " << scenariosPath);
        return allTrajectories;
    }
    
    // Parse all scenario directories
    for (const auto& entry : std::filesystem::directory_iterator (scenariosPath))
    {
        if (entry.is_directory ())
        {
            std::string scenarioId = entry.path ().filename ().string ();
            std::string xoscFile = entry.path () / (scenarioId + ".xosc");
            
            if (std::filesystem::exists (xoscFile))
            {
                NS_LOG_INFO ("Parsing scenario: " << scenarioId);
                auto trajectories = ParseOpenScenarioFile (xoscFile);
                allTrajectories[scenarioId] = trajectories;
                
                NS_LOG_INFO ("Scenario " << scenarioId << " contains " 
                            << trajectories.size () << " vehicles");
            }
        }
    }
    
    return allTrajectories;
}

std::vector<VehicleTrajectory> 
ShanghaiMobilityExtractor::ParseOpenScenarioFile (const std::string& filePath)
{
    NS_LOG_FUNCTION (filePath);
    
    std::vector<VehicleTrajectory> trajectories;
    
    // For now, create a mock implementation
    // TODO: Implement actual XML parsing with TinyXML-2
    
    // Mock data for demonstration
    VehicleTrajectory mockTrajectory;
    mockTrajectory.vehicleId = "mock_vehicle";
    mockTrajectory.scenarioId = "mock_scenario";
    
    // Create some mock waypoints
    for (int i = 0; i < 100; ++i)
    {
        double time = i * 0.1; // 100ms intervals
        double x = 38.0 + i * 0.5; // Moving in X direction
        double y = -52.0 + i * 0.2; // Moving in Y direction
        double z = 0.2;
        double heading = 4.14; // Constant heading
        
        TrajectoryPoint point(time, Vector(x, y, z), heading);
        mockTrajectory.waypoints.push_back (point);
    }
    
    // Calculate trajectory metrics
    if (!mockTrajectory.waypoints.empty ())
    {
        mockTrajectory.initialPosition = mockTrajectory.waypoints[0].position;
        mockTrajectory.finalPosition = mockTrajectory.waypoints.back ().position;
        mockTrajectory.duration = mockTrajectory.waypoints.back ().time;
        mockTrajectory.totalDistance = CalculateDistance (
            mockTrajectory.initialPosition, mockTrajectory.finalPosition);
        mockTrajectory.avgSpeed = mockTrajectory.totalDistance / mockTrajectory.duration;
    }
    
    trajectories.push_back (mockTrajectory);
    
    return trajectories;
}

std::vector<VehicleTrajectory> 
ShanghaiMobilityExtractor::SelectBest63UEs (const std::map<std::string, std::vector<VehicleTrajectory>>& allTrajectories)
{
    NS_LOG_FUNCTION ("Selecting best 63 UEs");
    
    std::vector<VehicleTrajectory> allCandidates;
    
    // Collect all candidate trajectories
    for (const auto& [scenarioId, trajectories] : allTrajectories)
    {
        for (const auto& traj : trajectories)
        {
            // Filter by minimum requirements
            if (traj.duration >= 10.0 && // At least 10 seconds
                traj.avgSpeed >= 1.0 && traj.avgSpeed <= 30.0 && // Realistic speeds
                traj.totalDistance >= 50.0) // At least 50 meters
            {
                allCandidates.push_back (traj);
            }
        }
    }
    
    NS_LOG_INFO ("Found " << allCandidates.size () << " candidate UEs");
    
    // Sort by score (combination of duration, speed, distance)
    std::sort (allCandidates.begin (), allCandidates.end (),
              [this](const VehicleTrajectory& a, const VehicleTrajectory& b)
              {
                  return CalculateTrajectoryScore (a) > CalculateTrajectoryScore (b);
              });
    
    // Select top 63, ensuring diversity
    std::vector<VehicleTrajectory> selected63;
    std::map<std::string, int> scenarioCount;
    
    for (const auto& traj : allCandidates)
    {
        if (selected63.size () >= 63)
            break;
            
        // Limit max 15 UEs per scenario for diversity
        if (scenarioCount[traj.scenarioId] < 15)
        {
            selected63.push_back (traj);
            scenarioCount[traj.scenarioId]++;
        }
    }
    
    NS_LOG_INFO ("Selected " << selected63.size () << " UEs from " 
                << scenarioCount.size () << " scenarios");
    
    return selected63;
}

void 
ShanghaiMobilityExtractor::ConvertToNs2Format (const std::vector<VehicleTrajectory>& trajectories, 
                                              const std::string& outputPath)
{
    NS_LOG_FUNCTION (outputPath);
    
    std::ofstream outFile (outputPath);
    if (!outFile.is_open ())
    {
        NS_LOG_ERROR ("Cannot open output file: " << outputPath);
        return;
    }
    
    // Write NS-2 mobility format
    for (size_t ueId = 0; ueId < trajectories.size (); ++ueId)
    {
        const auto& traj = trajectories[ueId];
        
        if (traj.waypoints.empty ())
            continue;
            
        // Initial position
        const auto& initial = traj.waypoints[0];
        outFile << "$node_(" << ueId << ") set X_ " << initial.position.x << std::endl;
        outFile << "$node_(" << ueId << ") set Y_ " << initial.position.y << std::endl;
        outFile << "$node_(" << ueId << ") set Z_ " << initial.position.z << std::endl;
        
        // Movement commands
        for (size_t i = 1; i < traj.waypoints.size (); ++i)
        {
            const auto& prev = traj.waypoints[i-1];
            const auto& curr = traj.waypoints[i];
            
            double distance = CalculateDistance (prev.position, curr.position);
            double speed = distance / (curr.time - prev.time);
            
            outFile << "$ns_ at " << prev.time 
                    << " \"$node_(" << ueId << ") setdest " 
                    << curr.position.x << " " << curr.position.y << " " << curr.position.z
                    << " " << speed << "\"" << std::endl;
        }
    }
    
    outFile.close ();
    NS_LOG_INFO ("NS-2 mobility file written to: " << outputPath);
}

void 
ShanghaiMobilityExtractor::SetupWaypointMobility (NodeContainer ueNodes, 
                                                   const std::vector<VehicleTrajectory>& trajectories)
{
    NS_LOG_FUNCTION ("Setting up waypoint mobility for " << ueNodes.GetN () << " UEs");
    
    if (ueNodes.GetN () != trajectories.size ())
    {
        NS_LOG_ERROR ("Number of UEs (" << ueNodes.GetN () 
                     << ") does not match number of trajectories (" << trajectories.size () << ")");
        return;
    }
    
    for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    {
        Ptr<Node> ueNode = ueNodes.Get (i);
        const auto& traj = trajectories[i];
        
        // Create waypoint mobility model
        Ptr<WaypointMobilityModel> mobility = CreateObject<WaypointMobilityModel> ();
        ueNode->AggregateObject (mobility);
        
        // Add all waypoints
        for (const auto& point : traj.waypoints)
        {
            Waypoint waypoint (Seconds (point.time), point.position);
            mobility->AddWaypoint (waypoint);
        }
        
        NS_LOG_INFO ("UE " << i << " (" << traj.vehicleId 
                    << ") configured with " << traj.waypoints.size () << " waypoints");
    }
}

double 
ShanghaiMobilityExtractor::CalculateDistance (const Vector& p1, const Vector& p2)
{
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dz = p2.z - p1.z;
    return std::sqrt (dx*dx + dy*dy + dz*dz);
}

double 
ShanghaiMobilityExtractor::CalculateTrajectoryScore (const VehicleTrajectory& traj)
{
    // Score based on duration, distance, and speed diversity
    double durationScore = std::min (traj.duration / 60.0, 1.0); // Max 1 minute = 1.0
    double distanceScore = std::min (traj.totalDistance / 500.0, 1.0); // Max 500m = 1.0
    double speedScore = 1.0 - std::abs (traj.avgSpeed - 15.0) / 15.0; // Prefer 15 m/s
    
    return 0.4 * durationScore + 0.4 * distanceScore + 0.2 * speedScore;
}

Vector 
ShanghaiMobilityExtractor::ConvertPosition (tinyxml2::XMLElement* positionElement)
{
    // TODO: Implement actual XML position conversion
    // For now, return mock position
    return Vector (0.0, 0.0, 0.0);
}

} // namespace ns3