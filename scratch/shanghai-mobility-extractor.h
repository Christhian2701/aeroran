#ifndef SHANGHAI_MOBILITY_EXTRACTOR_H
#define SHANGHAI_MOBILITY_EXTRACTOR_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include <vector>
#include <string>
#include <map>
#include <tinyxml2.h>

namespace ns3 {

/**
 * \brief Structure to represent a trajectory waypoint
 */
struct TrajectoryPoint
{
    double time;      // Timestamp in seconds
    Vector position;   // 3D position (x, y, z)
    double heading;    // Direction in radians
    
    TrajectoryPoint(double t, Vector pos, double h) 
        : time(t), position(pos), heading(h) {}
};

/**
 * \brief Structure to represent a vehicle trajectory
 */
struct VehicleTrajectory
{
    std::string vehicleId;
    std::string scenarioId;
    std::vector<TrajectoryPoint> waypoints;
    double totalDistance;
    double avgSpeed;
    double duration;
    Vector initialPosition;
    Vector finalPosition;
};

/**
 * \brief Extractor for Shanghai OpenSCENARIO mobility data
 */
class ShanghaiMobilityExtractor
{
public:
    /**
     * \brief Parse all scenarios in dataset directory
     * \param datasetPath Path to Shanghai dataset
     * \return Map of scenarioId to vehicle trajectories
     */
    std::map<std::string, std::vector<VehicleTrajectory>> ParseAllScenarios (const std::string& datasetPath);
    
    /**
     * \brief Parse single OpenSCENARIO file
     * \param filePath Path to .xosc file
     * \return Vector of vehicle trajectories
     */
    std::vector<VehicleTrajectory> ParseOpenScenarioFile (const std::string& filePath);
    
    /**
     * \brief Select best 63 UEs from all available trajectories
     * \param allTrajectories All parsed trajectories
     * \return Selected trajectories for 63 UEs
     */
    std::vector<VehicleTrajectory> SelectBest63UEs (const std::map<std::string, std::vector<VehicleTrajectory>>& allTrajectories);
    
    /**
     * \brief Convert trajectory to NS-2 mobility format
     * \param trajectories Vehicle trajectories
     * \param outputPath Output file path
     */
    void ConvertToNs2Format (const std::vector<VehicleTrajectory>& trajectories, const std::string& outputPath);
    
    /**
     * \brief Setup Waypoint mobility for UEs
     * \param ueNodes UE container
     * \param trajectories Vehicle trajectories  
     */
    void SetupWaypointMobility (NodeContainer ueNodes, const std::vector<VehicleTrajectory>& trajectories);

private:
    /**
     * \brief Calculate distance between two points
     */
    double CalculateDistance (const Vector& p1, const Vector& p2);
    
    /**
     * \brief Calculate trajectory score for UE selection
     */
    double CalculateTrajectoryScore (const VehicleTrajectory& traj);
    
    /**
     * \brief Convert OpenSCENARIO position to NS-3 Vector
     */
    Vector ConvertPosition (tinyxml2::XMLElement* positionElement);
};

} // namespace ns3

#endif /* SHANGHAI_MOBILITY_EXTRACTOR_H */