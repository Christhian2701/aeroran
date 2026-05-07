/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Shanghai Mobility Manager - Coordinates mobility for 63 UEs
 * Supports both WaypointMobilityModel and Ns2MobilityHelper approaches
 */

#ifndef SHANGHAI_MOBILITY_MANAGER_H
#define SHANGHAI_MOBILITY_MANAGER_H

#include "shanghai-ue-selector.h"
#include "shanghai-openscenario-parser.h"
#include "ns3/object.h"
#include "ns3/node-container.h"
#include "ns3/mobility-helper.h"
#include "ns3/waypoint-mobility-model.h"
#include "ns3/ns2-mobility-helper.h"
#include <vector>
#include <string>

namespace ns3 {

enum MobilityImplementationType
{
  WAYPOINT_MODEL = 0,
  NS2_MOBILITY_HELPER = 1,
  HYBRID_APPROACH = 2
};

struct MobilityConfiguration
{
  MobilityImplementationType implementation;
  double timeScale;                    // Scale factor for trajectory playback
  double initialDelay;                 // Delay before starting mobility (seconds)
  bool enableCourseChangeTracing;      // Log position changes
  std::string traceFilePath;           // Path for mobility trace files
  double waypointUpdateInterval;       // Update interval for waypoint models (seconds)
  bool synchronizeStartTime;           // Sync all UEs to start simultaneously
};

struct UEAssignment
{
  int nodeId;
  std::string scenarioId;
  std::string vehicleId;
  VehicleInfo trajectory;
  double startTime;
  double endTime;
  Vector initialPosition;
};

class ShanghaiMobilityManager : public Object
{
public:
  static TypeId GetTypeId (void);
  ShanghaiMobilityManager ();
  virtual ~ShanghaiMobilityManager ();

  // Configuration
  void SetMobilityConfiguration (const MobilityConfiguration& config);
  void LoadUEAssignments (const std::vector<SelectedUE>& selectedUEs);
  
  // Implementation methods
  void SetupWaypointMobility (NodeContainer ueNodes);
  void SetupNs2Mobility (NodeContainer ueNodes, const std::string& ns2TraceFile);
  void SetupHybridMobility (NodeContainer ueNodes, const std::string& ns2TraceFile);
  
  // Advanced features
  void EnableRealTimeVisualization (const std::string& outputPath);
  void ValidateMobilitySetup (NodeContainer ueNodes);
  void ExportMobilitySummary (const std::string& summaryPath);
  
  // Runtime control
  void StartMobility ();
  void StopMobility ();
  void PauseMobility ();
  void ResumeMobility ();
  void SetTimeScale (double scale);
  
  // Statistics and monitoring
  std::vector<Vector> GetCurrentPositions (NodeContainer ueNodes) const;
  std::vector<Vector> GetCurrentVelocities (NodeContainer ueNodes) const;
  void LogMobilityStatistics (NodeContainer ueNodes) const;

private:
  // Waypoint model specific methods
  void ConfigureWaypointForUE (Ptr<Node> ue, const UEAssignment& assignment);
  std::vector<Waypoint> ConvertTrajectoryToWaypoints (const VehicleInfo& vehicle, 
                                                      double timeScale,
                                                      double initialDelay);
  
  // NS-2 mobility specific methods
  void GenerateNs2TraceFile (const std::string& outputPath);
  void ConfigureNs2Helper (Ns2MobilityHelper& ns2Helper);
  
  // Utility methods
  Vector TransformCoordinates (const Vector& originalPos, const std::string& scenarioId) const;
  void OptimizeTrajectories (std::vector<UEAssignment>& assignments);
  void ValidateTrajectoryData (const VehicleInfo& vehicle) const;
  
  // Event handling
  void ScheduleMobilityUpdates ();
  void HandleCourseChange (Ptr<const MobilityModel> model);
  
  MobilityConfiguration m_config;
  std::vector<UEAssignment> m_ueAssignments;
  std::map<int, UEAssignment> m_nodeAssignmentMap;
  bool m_mobilityStarted;
  EventId m_updateEvent;
  double m_simulationStartTime;
  
  // Statistics
  uint32_t m_totalWaypoints;
  double m_totalSimulationTime;
  uint32_t m_activeConnections;
};

// Factory functions for different approaches
Ptr<ShanghaiMobilityManager> CreateWaypointMobilityManager (const MobilityConfiguration& config);
Ptr<ShanghaiMobilityManager> CreateNs2MobilityManager (const MobilityConfiguration& config, 
                                                       const std::string& ns2File);
Ptr<ShanghaiMobilityManager> CreateHybridMobilityManager (const MobilityConfiguration& config);

} // namespace ns3

#endif /* SHANGHAI_MOBILITY_MANAGER_H */